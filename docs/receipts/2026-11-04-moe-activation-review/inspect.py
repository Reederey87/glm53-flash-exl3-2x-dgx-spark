from pathlib import Path
from collections import Counter
import hashlib,json,datetime,re
revision='76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'
root=Path('/root/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots')/revision
raw=(root/'config.json').read_bytes();sha=hashlib.sha256(raw).hexdigest()
assert sha=='dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8'
config=json.loads(raw);q=config['quantization_config']['quantized_layers'];roles=Counter();experts={}
for name,value in q.items():
    algo=value['quant_algo'];role='routed_experts' if '.experts' in name else 'shared_expert' if 'shared_expert' in name else 'other'
    roles[role+':'+algo]+=1
    if role=='routed_experts':experts[name]=algo
controls=[]
allowed=('VLLM_B12X_MOE_FP4_FORCE_A16','B12X_W4A16_A4_PREFILL','B12X_W4A16_A4_PREFILL_TERMS','VLLM_B12X_ACTIVATION_MODE_A16_M')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        if b'serve' not in argv or b'/usr/local/bin/vllm' not in argv:continue
        env=dict(x.decode().split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
        controls.append({k:env.get(k,'unset') if k not in env or re.fullmatch(r'[0-9]+',env[k]) else 'non-numeric-redacted' for k in allowed})
    except (OSError,UnicodeDecodeError):continue
assert len(controls)==1 and controls[0]['VLLM_B12X_MOE_FP4_FORCE_A16']=='1' and controls[0]['B12X_W4A16_A4_PREFILL']=='1'
assert roles['routed_experts:NVFP4']==42 and roles['routed_experts:W4A16_NVFP4']==1
site=Path('/usr/local/lib/python3.12/dist-packages')
files=('vllm/model_executor/layers/quantization/nvfp4_csf.py','vllm/model_executor/layers/quantization/modelopt.py','vllm/model_executor/layers/fused_moe/b12x.py','vllm/utils/b12x.py','flashinfer/experimental/b12x/moe/fused_moe/planning.py','flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/prefill_a4.py')
print(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':revision,'config_sha256':sha,'controls':controls[0],'quantization_roles':dict(roles),'routed_expert_layers':experts,'text_shape':{k:config['text_config'][k] for k in ('hidden_size','moe_intermediate_size','n_routed_experts','num_experts_per_tok','num_hidden_layers','swiglu_limit')},'installed_source_sha256':{p:hashlib.sha256((site/p).read_bytes()).hexdigest() for p in files},'scope':'Read-only source/public metadata/whitelisted numeric controls; no GPU calls or live plan introspection.'},indent=2))
