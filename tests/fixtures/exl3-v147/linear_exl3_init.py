# Compact v1.4.7 LinearEXL3 construction contract for host tests.
# Real LinearEXL3 does this before reading config.infer_params in forward().

from ...model.config import Config, NullConfig  # noqa: F401


def construct(config=None):
    if config is None:
        from ...model.config import NullConfig as _NullConfig
        config = _NullConfig()
    return config.infer_params.no_reconstruct
