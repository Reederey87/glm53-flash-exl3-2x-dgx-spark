"""Prepared SM121 native-A4 gate; never run alongside inference.

Synthetic packed weights use the real GLM TP2 shard dimensions. Every row and
route is compared with the pinned original-layout Torch oracle. This is not
checkpoint quality, CSF storage qualification or automatic dispatch proof.
"""
from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path('/opt/moe-native-a4-tests')
CORPUS = 'tests/experimental/b12x/moe/test_cute_migration_moe_standard_corpus.py'
CORPUS_SHA = '45e71d0dca05dbf5487f1ef948b5b31b863bba5e7c1b834c420beb6621397dca'
REFERENCE_SHA = 'e7b34369810c95b2ed469bd4305e224e9e83843414fc69af100b33d9eb96b8a4'
HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 4096, 1024, 288, 8
LIMIT = 10.0
MIN_COS, MAX_NORMALIZED_RMSE = 0.9999, 0.015

# Compact support ends at eight live tokens in the pinned kernel. M24 is an
# MTP verification case for the dynamic path, not a supported compact case.
CASES = (
    [(m, 'micro', False, 'nonuniform') for m in (1, 4, 6, 8)]
    + [(m, 'dynamic', fast, 'nonuniform')
       for m in (1, 4, 6, 8, 9, 16, 17, 24, 32, 33, 4096)
       for fast in (False, True)]
    + [(m, backend, fast, profile)
       for profile in ('uniform', 'subnormal')
       for m, backend, fast in ((6, 'micro', False),
                                (33, 'dynamic', False), (33, 'dynamic', True))]
)


def _case_id(case):
    m, backend, fast, profile = case
    return f'm{m}-{backend}-fast{int(fast)}-{profile}'


def _sources():
    source = ROOT / CORPUS
    assert hashlib.sha256(source.read_bytes()).hexdigest() == CORPUS_SHA
    sys.path.insert(0, str(ROOT))
    corpus = importlib.import_module('tests.experimental.b12x.moe.test_cute_migration_moe_standard_corpus')
    assert Path(corpus.__file__).resolve() == source, 'unexpected fixture import owner'
    reference = importlib.import_module('b12x.moe._shared.kernels.reference')
    assert hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() == REFERENCE_SHA
    assert ROOT not in Path(reference.__file__).resolve().parents, 'test tree must not supply kernels'
    return corpus, reference


def _require_room(torch, requested_bytes):
    """Refuse a large allocation before touching unified memory."""
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    free = int(meminfo['MemFree'].split()[0]) * 1024
    assert free >= requested_bytes + 24 * 1024**3, 'insufficient free memory plus native safety floor'
    assert torch.cuda.memory_allocated() + requested_bytes < 32 * 1024**3, 'planned tensor footprint exceeds bound'


def _scratch(torch, specs, device):
    specs = tuple(specs)
    requested = sum(math.prod(s.shape) * s.dtype.itemsize for s in specs)
    _require_room(torch, requested)
    return tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in specs)


def _scale_summary(torch, values, global_scale, math_mode):
    """Observe the reference's block-scale rounding, not an inferred profile."""
    maxima = values.float().reshape(-1, 16).abs().amax(dim=-1)
    gs = global_scale.float()
    if math_mode == 'micro':
        raw = (maxima.double() * gs.double() / 6.0).float()
    elif math_mode == 'dynamic_fast':
        sixth = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=values.device)
        raw = (maxima * sixth) * gs
    else:
        raw = (maxima * gs) / 6.0
    scales = raw.clamp(max=448.0).to(torch.float8_e4m3fn).float()
    return {'zero': int((scales == 0).sum().item()),
            'subnormal': int(((scales > 0) & (scales < 2.0**-6)).sum().item()),
            'normal': int((scales >= 2.0**-6).sum().item())}


@pytest.mark.parametrize('case', CASES, ids=_case_id)
def test_glm_tp2_native_a4_oracle_and_live_replay(case, record_property):
    import torch
    from b12x.moe import fused_moe
    from b12x.preparation import PreparationSession, PreparedCall

    assert torch.cuda.is_available(), 'requires CUDA; do not count a skip as qualification'
    assert torch.cuda.get_device_capability() == (12, 1), 'requires a GB10 SM121 device'
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    corpus, reference = _sources()
    assert torch.get_float32_matmul_precision() == 'highest', 'Torch oracle must not use reduced-precision matmul'
    rows, backend, fast_math, profile = case
    math_mode = 'micro' if backend == 'micro' else ('dynamic_fast' if fast_math else 'dynamic_precise')
    _require_room(torch, 8 * 1024**3)
    weights = corpus._make_nvfp4_weights(
        device, seed=2026110401, num_experts=EXPERTS,
        hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
    )
    if profile == 'subnormal':
        a1 = torch.linspace(2.0**-10, 2.0**-9, EXPERTS, device=device)
        a2 = torch.linspace(2.0**-12, 2.0**-11, EXPERTS, device=device)
    else:
        a1 = torch.linspace(16.0, 32.0, EXPERTS, device=device)
        a2 = torch.linspace(8.0, 16.0, EXPERTS, device=device)
        if profile == 'uniform':
            a1.fill_(32.0)
            a2.fill_(16.0)
    # Physical weight scales stay fixed when activation calibration changes.
    global1, global2 = weights.w1_alpha, weights.w2_alpha
    weights = replace(weights, a1_scale=a1, a2_scale=a2,
                      w1_alpha=weights.w1_alpha / a1,
                      w2_alpha=weights.w2_alpha / a2)
    initial = corpus._make_inputs(device, m=rows, seed=2026110402, route_shift=0,
                                  num_experts=EXPERTS, hidden_size=HIDDEN, topk=TOPK)
    changed = corpus._make_inputs(device, m=rows, seed=2026110403, route_shift=97,
                                  num_experts=EXPERTS, hidden_size=HIDDEN, topk=TOPK)
    # Exercise clamp behavior rather than merely passing a dormant limit.
    initial.a.mul_(8.0)
    changed.a.mul_(8.0)
    if rows > 1:
        changed.topk_ids[-1].fill_(-1)

    def oracle(inputs):
        valid = inputs.topk_ids >= 0
        ids = inputs.topk_ids.clamp(min=0)
        probabilities = torch.where(valid, inputs.topk_weights, 0.0)
        return reference.moe_reference_nvfp4(
            inputs.a, weights.w1_fp4, weights.w1_scale, weights.w1_alpha,
            weights.w2_fp4, weights.w2_scale, weights.w2_alpha,
            a1, a2, ids, probabilities, EXPERTS, HIDDEN, INTERMEDIATE,
            activation='silu', swiglu_limit=LIMIT, quant_scale_math=math_mode,
        )

    trace = reference.trace_moe_reference_nvfp4_route(
        initial.a, weights.w1_fp4, weights.w1_scale, weights.w1_alpha,
        weights.w2_fp4, weights.w2_scale, weights.w2_alpha, a1, a2,
        initial.topk_ids, initial.topk_weights, EXPERTS, HIDDEN, INTERMEDIATE,
        token_idx=0, route_idx=0, activation='silu', swiglu_limit=LIMIT,
        quant_scale_math=math_mode,
    )
    assert bool(((trace.gate_out > LIMIT) | (trace.up_out.abs() > LIMIT)).any().item()), 'clamp fixture is dormant'
    eid = trace.expert_idx
    observed_scales = {
        'input': _scale_summary(torch, initial.a[0], a1[eid], math_mode),
        'intermediate': _scale_summary(torch, trace.intermediate, a2[eid], math_mode),
    }
    if profile == 'subnormal':
        assert all(item['subnormal'] > 0 for item in observed_scales.values()), 'subnormal fixture is dormant'
    del trace
    # This deliberately uses the original oracle for EVERY row and route,
    # including M4096. It is slow but holds only one route's dense weights.
    initial_reference, changed_reference = oracle(initial), oracle(changed)
    assert not torch.equal(initial_reference, changed_reference), 'changed-input oracle is unchanged'
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format='modelopt_nvfp4', w13_layout='w13'),
        activation=fused_moe.ActivationSpec(mode='a4', nonlinearity='silu',
                                           io_dtype=torch.bfloat16, swiglu_limit=LIMIT),
        geometry=fused_moe.MoEGeometry(num_experts=EXPERTS, hidden_size=HIDDEN,
                                      intermediate_size=INTERMEDIATE),
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=weights.w1_fp4, w2=weights.w2_fp4,
        w13_block_scales=weights.w1_scale, w2_block_scales=weights.w2_scale,
        w13_global_scales=global1, w2_global_scales=global2,
        input_scale=a1, intermediate_scale=a2, immutable_input_scales=True,
    ))
    override = fused_moe.MoeDecodeConfig(
        backend=backend, route_planner='internal', max_active_clusters=None,
        dynamic_tile_m=16 if backend == 'dynamic' else None,
        dynamic_route_mode='grouped' if backend == 'dynamic' else None,
        nvfp4_share_input=False,
    )
    plan = fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(max_tokens=rows, top_k=TOPK),
        invocation={'fast_math': fast_math}, override=override,
    )

    def prepare(state):
        scratch = _scratch(torch, state.scratch.scratch_specs(), device)
        output = torch.empty_like(initial.a)
        binding = state.bind(a=initial.a, topk_ids=initial.topk_ids,
                             topk_weights=initial.topk_weights, scratch=scratch,
                             output=output, input_scales_static=True)
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    measurements = []

    def check(actual, expected, phase):
        corpus._assert_oracle(actual, expected, context=f'{_case_id(case)}:{phase}',
                              min_cos=MIN_COS, max_normalized_rmse=MAX_NORMALIZED_RMSE)
        result = reference.compare_to_reference(actual, expected)
        rms = expected.float().square().mean().sqrt().item()
        measurements.append({'phase': phase, 'cos': result.cos, 'max_abs': result.max_abs,
                             'normalized_rmse': result.rmse / rms})

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare((plan.request(name='glm-tp2-native-a4', prepare_call=prepare),))
        assert plan.prepared.state.config.backend == backend, 'wrong selected backend'
        assert plan.prepared.state.scratch.caps.w4a16_fast_math == fast_math
        scratch = _scratch(torch, plan.scratch_specs(), device)
        # A sentinel row outside the bound output catches an overrun.
        output = torch.empty((rows + 1, HIDDEN), dtype=torch.bfloat16, device=device)
        output.fill_(float('nan'))
        binding = fused_moe.bind(plan, a=initial.a, topk_ids=initial.topk_ids,
                                 topk_weights=initial.topk_weights, scratch=scratch,
                                 output=output[:rows], input_scales_static=True)
        assert binding.implementation == backend
        session.freeze()
        binding.run()
        torch.cuda.synchronize(device)
        check(output[:rows], initial_reference, 'eager')
        assert torch.isnan(output[rows:]).all()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                binding.run()
            torch.cuda.synchronize(device)
            check(output[:rows], initial_reference, 'capture')
            initial.a.copy_(changed.a)
            initial.topk_ids.copy_(changed.topk_ids)
            initial.topk_weights.copy_(changed.topk_weights)
            live = (*scratch, output, initial.a, initial.topk_ids, initial.topk_weights)
            addresses = tuple(t.data_ptr() for t in live)
            for replay in range(3):
                output.fill_(float('nan'))
                if binding.barrier_count is not None:
                    binding.barrier_count.fill_(123)
                    binding.barrier_epoch.fill_(-42)
                before = torch.cuda.memory_stats(device)['allocation.all.allocated']
                live_bytes = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_stats(device)['allocation.all.allocated'] == before
                assert torch.cuda.memory_allocated(device) == live_bytes
                assert tuple(t.data_ptr() for t in live) == addresses
                check(output[:rows], changed_reference, f'replay-{replay}')
                assert torch.isnan(output[rows:]).all()
                if rows > 1:
                    assert torch.count_nonzero(output[rows - 1]).item() == 0
        finally:
            graph.reset()
    peak = torch.cuda.max_memory_allocated(device)
    assert peak < 48 * 1024**3, 'native tensor allocation exceeded the registered component bound'
    record_property('native_a4_evidence', json.dumps({
        'case': _case_id(case), 'geometry': [rows, HIDDEN, INTERMEDIATE, EXPERTS, TOPK],
        'quant_scale_math': math_mode, 'swiglu_limit': LIMIT, 'all_rows_oracled': True,
        'torch_float32_matmul_precision': torch.get_float32_matmul_precision(),
        'observed_first_route_scale_classes': observed_scales, 'measurements': measurements,
        'peak_allocated_bytes': peak, 'corpus_sha256': CORPUS_SHA, 'reference_sha256': REFERENCE_SHA,
    }, sort_keys=True))
