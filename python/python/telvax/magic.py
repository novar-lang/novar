"""``%%telvax``: Telvax cells inside a Python notebook (or IPython).

    %load_ext telvax

    %%telvax
    fn double(n: Int) -> Int:
        n * 2

    %%telvax
    double(21)

Definitions and variables last from one cell to the next, as in ``telvax repl``. The session belongs
to the notebook's IPython shell; ``%telvax_reset`` forgets it.
"""

from __future__ import annotations

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.display import display

from . import Session


@magics_class
class TelvaxMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.session = Session()

    @cell_magic("telvax")
    def telvax(self, line, cell):
        """Runs the cell as Telvax and shows the value of its last expression."""
        result = self.session.eval(cell)
        if result["output"]:
            print(result["output"], end="")
        if result["error"]:
            raise UsageError(result["error"])
        value = result["value"]
        if value is not None:
            data = {"text/plain": value["text"]}
            if value["html"]:
                data["text/html"] = value["html"]
            display(data, raw=True)
        for note in result["notes"]:
            print(f"note: {note}")

    @line_magic("telvax_reset")
    def telvax_reset(self, line):
        """Forgets every Telvax definition and variable of this notebook."""
        self.session.reset()

    @line_magic("telvax_vars")
    def telvax_vars(self, line):
        """Lists the Telvax variables as ``name: type = value``."""
        for name, ty, value in self.session.variables():
            print(f"{name}: {ty} = {value}")


def load_ipython_extension(ipython):
    ipython.register_magics(TelvaxMagics)
