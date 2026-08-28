#!/usr/bin/env python3
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/model.py")
text = p.read_text()
old = '            overrides = [\n                "auto_gptq",\n'
new = '            overrides = [\n                "exl3",\n                "auto_gptq",\n'
if text.count(old) != 1:
    raise SystemExit("overrides list target missing or not unique")
p.write_text(text.replace(old, new))
print("exl3 added to ModelConfig overrides")
