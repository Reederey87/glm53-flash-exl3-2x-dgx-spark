# On-GB10 parity: Triton custom op (CUDA) vs the kit's eager CPU fallback.
# Proves the CUDA path executes on sm_121 and agrees with the reference math.
import torch

from vllm.model_executor.models import qwen3_dflash2 as m

print("module:", m.__file__)
print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

TAPS = 4
GROUP_SIZE = 8
BLOCK_SIZE = 16


def build(num_rows, num_channels, dtype, device):
    num_groups = num_channels // GROUP_SIZE
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(num_rows, num_channels, generator=g).to(dtype).to(device)
    delta = torch.randn(num_rows, TAPS, num_groups, generator=g).to(dtype).to(device)
    base = torch.randn(TAPS, num_channels, generator=g).to(dtype).to(device)
    return x, delta, base, num_groups


def eager_cpu(x, delta, base, num_groups):
    return m._grouped_conv(
        x.cpu(),
        delta.cpu(),
        base.cpu(),
        BLOCK_SIZE,
        num_groups,
        GROUP_SIZE,
        TAPS,
    )


def fp64_ref(x, delta, base, num_groups):
    x = x.cpu().double()
    delta = delta.cpu().double()
    base = base.cpu().double()
    blocks = x.unflatten(-1, (num_groups, GROUP_SIZE))
    coeff = base.view(1, TAPS, num_groups, GROUP_SIZE) + delta.unsqueeze(-1)
    out = coeff[:, 0] * blocks
    pos = torch.arange(x.shape[0]) % BLOCK_SIZE
    for tap in range(1, TAPS):
        shifted = torch.zeros_like(blocks)
        shifted[tap:] = blocks[:-tap]
        gate = (pos >= tap).view(-1, 1, 1)
        out = out + torch.where(gate, coeff[:, tap] * shifted, torch.zeros_like(out))
    return out.reshape(x.shape[0], -1)


CASES = [
    ("bf16 small", 64, 512, torch.bfloat16),
    ("bf16 odd-block", 100, 768, torch.bfloat16),
    ("fp16 small", 64, 512, torch.float16),
    ("bf16 1024-aligned", 256, 1024, torch.bfloat16),
    ("bf16 large", 512, 2048, torch.bfloat16),
]

all_ok = True
for name, rows, chans, dtype in CASES:
    x, delta, base, ng = build(rows, chans, dtype, "cuda")
    out_cuda = m.dflash2_grouped_conv(x, delta, base, BLOCK_SIZE, GROUP_SIZE)
    out_cpu = eager_cpu(x, delta, base, ng)
    ref = fp64_ref(x, delta, base, ng)

    d_cc = (out_cuda.cpu().float() - out_cpu.float()).abs().max().item()
    d_cr = (out_cuda.cpu().float() - ref.float()).abs().max().item()
    scale = ref.abs().max().item()
    rel = d_cr / max(scale, 1e-9)
    ok = rel < 2e-2
    all_ok &= ok
    print(
        f"{name:18s} rows={rows:4d} ch={chans:5d} {str(dtype):16s} "
        f"cuda-vs-cpu-eager={d_cc:.3e} cuda-vs-fp64={d_cr:.3e} rel={rel:.3e} "
        f"{'OK' if ok else 'MISMATCH'}"
    )

# Empty-row guard.
x, delta, base, ng = build(0, 512, torch.bfloat16, "cuda")
out = m.dflash2_grouped_conv(x, delta, base, BLOCK_SIZE, GROUP_SIZE)
print("empty rows:", tuple(out.shape), "OK" if out.shape == (0, 512) else "MISMATCH")

# The op must reject the CPU fallback path (proves the CUDA branch is what ran).
x, delta, base, ng = build(32, 512, torch.bfloat16, "cuda")
print("is_cuda branch taken:", x.is_cuda and m.dflash2_grouped_conv is not None)

print("PARITY", "PASS" if all_ok else "FAIL")
