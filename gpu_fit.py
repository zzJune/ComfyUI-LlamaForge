"""Shared GPU fit-target policy (MiB)."""

DEFAULT_FIT_TARGET = 1024
MIN_FIT_TARGET = 128
MAX_FIT_TARGET = 65536


def validate_fit_target(value):
    if type(value) is not int or not MIN_FIT_TARGET <= value <= MAX_FIT_TARGET:
        raise ValueError(f"LlamaForge: fit_target must be an integer from {MIN_FIT_TARGET} to {MAX_FIT_TARGET} MiB.")
    return value
