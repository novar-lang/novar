"""Run `std::dist::map` on a Ray cluster.

    import ray, novar
    ray.init()                                   # or ray.init(address="auto") on a cluster
    program = novar.compile(SOURCE)
    program.set_executor(novar.ray.RayExecutor(SOURCE))
    program.call("process_all", items)           # each mapped call becomes a Ray task

Every Ray worker compiles the program once and then runs the mapped function on the items it is
given. Items and results travel as JSON, converted with the function's static types, exactly as
on the built-in cluster (`novar worker`).
"""

import hashlib
import json

import ray

_PROGRAMS = {}


def _program(source):
    """Compiles `source` once per worker process."""
    import novar

    key = hashlib.sha256(source.encode()).hexdigest()
    if key not in _PROGRAMS:
        _PROGRAMS[key] = novar.compile(source)
    return _PROGRAMS[key]


@ray.remote
def _run_chunk(source, function, items):
    program = _program(source)
    return [program.call_json(function, item) for item in items]


class RayExecutor:
    """An executor for `Program.set_executor` that fans calls out as Ray tasks.

    `chunk` items are grouped per task so that cheap functions do not drown in scheduling
    overhead; `None` picks about four tasks per available CPU.
    """

    def __init__(self, source, chunk=None):
        self.source = source
        self.chunk = chunk

    def __call__(self, function, items):
        if not items:
            return []
        size = self.chunk
        if size is None:
            cpus = max(1, int(ray.cluster_resources().get("CPU", 1)))
            size = max(1, -(-len(items) // (cpus * 4)))
        refs = [
            _run_chunk.remote(self.source, function, items[i : i + size])
            for i in range(0, len(items), size)
        ]
        return [result for chunk in ray.get(refs) for result in chunk]
