"""Telvax: call the language from Python and exchange tensors and tables without copying."""

from ._native import (  # noqa: F401
    CompileError,
    DeviceTensor,
    TelvaxError,
    Program,
    Session,
    StatusError,
    Table,
    Tensor,
    __version__,
    compile,
    load,
    table,
    tensor,
    version,
)

__all__ = [
    "CompileError", "DeviceTensor", "TelvaxError", "Program", "Session", "StatusError", "Table", "Tensor",
    "compile", "load", "table", "tensor", "version", "ray",
]


def load_ipython_extension(ipython):
    """``%load_ext telvax`` adds the ``%%telvax`` cell magic."""
    from .magic import load_ipython_extension as load

    load(ipython)


def __getattr__(name):
    # `telvax.ray` imports Ray, which is optional and slow to import: load it on first use.
    if name == "ray":
        from . import ray

        return ray
    raise AttributeError(f"module 'telvax' has no attribute {name!r}")
