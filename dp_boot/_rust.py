"""Lazy access to the separately installed, optional native sampler."""
from importlib import import_module


def native():
    try:
        return import_module("dp_boot_native")
    except ModuleNotFoundError as exc:
        if exc.name != "dp_boot_native":
            raise
        raise ImportError(
            "backend='rust' requires the optional extension: "
            "install a Rust toolchain, then run python -m pip install ./rust"
        ) from exc


def mean_draws(values, replicates, rng):
    return native().mean_draws(values, replicates, rng.bytes(32))


def mean_counts(values, counts, replicates, rng):
    return native().mean_counts(values, counts, replicates, rng.bytes(32))
