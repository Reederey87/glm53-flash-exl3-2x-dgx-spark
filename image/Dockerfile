# GLM-5.3-Flash-NVFP4 runtime for 2x NVIDIA DGX Spark (GB10 / SM121) at TP=2.
#
# WHAT THIS ADDS
# The base image below already solves running GLM-5.3-Flash's NoPE sparse MLA on
# a SINGLE GB10: it pads the NoPE query into the 656-byte GLM_NSA cache ABI and
# rebuilds FlashInfer's AOT sparse module with a (64, 2176) decode
# specialization. All credit for that work to:
#     https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10
#
# It does NOT work at TP=2, which is what two Sparks require: GLM has 64
# attention heads, so each rank owns 32, FlashInfer's (num_heads, topk) decode
# lookup misses, and the kernel aborts with
#     Check failed: num_tokens > 64 (4 vs. 64)
# This image adds the missing (32, 2176) specialization. See image/patch_tp2.py
# and docs/03-tp2-kernel-fix.md.
#
# We build FROM their published image rather than vendoring their source: that
# repository ships no LICENSE, so redistributing its files is not permitted.
# Building on a public image is fine and keeps the attribution honest.
ARG BASE_IMAGE=ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10@sha256:4251b561d111d817765ed4097512ce36811deac071a4a7411d20242df5c74a47
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GLM-5.3-Flash-NVFP4 runtime for 2x DGX Spark (TP=2)" \
      org.opencontainers.image.description="Adds the FlashInfer (32,2176) sparse-MLA decode specialization required to run GLM-5.3-Flash across two GB10s at tensor-parallel 2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.glm53.base-image="ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10" \
      io.glm53.decode-specializations="32x2176,64x2176"

COPY patch_tp2.py verify_tp2.py /opt/glm53-tp2/

# Source edits first. Asserts every anchor, so a changed base fails here.
RUN python3 /opt/glm53-tp2/patch_tp2.py

# FlashInfer ships this module as a precompiled AOT .so which takes precedence
# over the edited sources -- editing Python/CUDA alone changes nothing. Rebuild
# for sm_121a and install back into the same AOT slot. This is the slow step
# (tens of minutes on a GB10).
RUN mv /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so \
       /opt/glm53-tp2/sparse_mla_sm120.base.so \
    && FLASHINFER_CUDA_ARCH_LIST=12.1a python3 -c \
       'from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build()' \
    && install -m 0755 \
       /root/.cache/flashinfer/0.6.17/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so \
       /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so

# Refuse to ship an image whose rebuilt module lacks either specialization.
RUN GLM53_GB10_PATCH_DISABLE=1 python3 /opt/glm53-tp2/verify_tp2.py

# Base ENTRYPOINT ["vllm", "serve"] is preserved.
