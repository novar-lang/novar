"""Novar: call the language from Python and exchange tensors and tables without copying."""

from ._native import (  # noqa: F401
    CompileError,
    DeviceTensor,
    NovarError,
    Program,
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
    "CompileError", "DeviceTensor", "NovarError", "Program", "StatusError", "Table", "Tensor",
    "compile", "load", "table", "tensor", "version", "ray",
]


def __getattr__(name):
    # `novar.ray` imports Ray, which is optional and slow to import: load it on first use.
    if name == "ray":
        from . import ray

        return ray
    raise AttributeError(f"module 'novar' has no attribute {name!r}")
