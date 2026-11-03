# Common runner dependency profile

`constraints.txt` records the 218 effective dependency versions from qualified
control image `40f4c5229d276c1d86c983389bc7281dbbe66b627b6d065a0d2f93bd1c405ab5`.
The complete effective inventory and raw-inventory hash are retained in
`control-effective-packages.json`. The raw inventory contains 223 distributions:
both OS setuptools 68.1.2 and effective local setuptools 78.1.0 are present.
Only the effective version belongs in the resolver constraints.

The three source-build context generators verify the profile hash before
creating a context, copy it into the runner and constrain its final two uv
solves. Compiler stages and existing Torch/DSL overrides remain unchanged.
Source-built vLLM and three FlashInfer wheels are excluded from this version
profile; their exact exported bytes are verified separately.

This prevents the observed OpenAI SDK drift from 3.43.0 to 3.44.0 between
control and candidate. It also retains cuDNN frontend 1.31.0 and DSL 4.7.0.
Constraints restrict resolution; they do not install or remove packages.
Require equality of the complete raw Python and Debian inventories after
building both arms, including shadowed distributions. A passing resolver alone
does not establish an identical environment or GPU qualification.

Before this change, Exa research reviewed uv's
[constraints](https://docs.astral.sh/uv/pip/compile/#dependency-constraints)
and [compatibility/overrides](https://docs.astral.sh/uv/pip/compatibility/).
A fully hashed offline wheelhouse is stronger but requires provenance for
every third-party artifact; this profile is not such a lock. `uv pip sync`
would change the inherited OS/local environment and is not used here.
Pinned vLLM admits `openai >= 2.25.0`; its x86_64 test lock is not an ARM64
production lock. Local Exllamav3 minimum bounds do not freeze a paired build.
TensorFold's CUDA helper checks device architecture and reports build-lock
waits, but does not resolve this package drift. No server or quant substitution
is involved.
