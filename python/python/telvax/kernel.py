"""A Jupyter kernel for Telvax: cells are Telvax, definitions and variables last from one cell to
the next, and the value of the last expression is shown (a table or a tensor as HTML).

    pip install "telvax[kernel]"
    jupyter lab                          # choose the Telvax kernel
    python -m telvax.kernel install      # registers the kernel for this user, if pip did not

It is the session of ``telvax repl`` behind the Jupyter messaging protocol, which ``ipykernel``
speaks.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

try:
    from ipykernel.kernelbase import Kernel
except ImportError as e:  # pragma: no cover - the message is the point
    raise ImportError(
        'the Telvax kernel needs ipykernel: pip install "telvax[kernel]"'
    ) from e

from . import Session, __version__

KERNEL_NAME = "telvax"

HELP = """\
Cells hold Telvax: a definition (fn, type, enum, trait, extend, import), a variable (let, var), a
statement or an expression, whose value is shown. Names last until :reset.

  :type <expression>   the type of an expression, without running it
  :vars                the variables, with their types and values
  :defs                the functions, types and the like defined so far
  :reset               forget everything
  :help                this text
Tab completes names, fields and methods; Shift+Tab shows the type of the name at the cursor."""


def _fence(markdown: str) -> str:
    """The text of a hover, without its code fences."""
    return re.sub(r"```\w*\n?", "", markdown).strip()


class TelvaxKernel(Kernel):
    implementation = "Telvax"
    implementation_version = __version__
    language_info = {
        "name": "telvax",
        "version": __version__,
        "mimetype": "text/x-telvax",
        "file_extension": ".tvx",
        # Telvax reads like Python: until there is a mode of its own, highlight it as Python.
        "codemirror_mode": {"name": "python", "version": 3},
        "pygments_lexer": "python3",
    }
    banner = f"Telvax {__version__}"
    help_links = [
        {"text": "Telvax", "url": "https://github.com/telvax/telvax"},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.repl = Session()

    # ---- running cells ---------------------------------------------------------------------

    def _stream(self, name: str, text: str) -> None:
        if text:
            self.send_response(self.iopub_socket, "stream", {"name": name, "text": text})

    def _failure(self, message: str, silent: bool) -> dict:
        lines = message.splitlines() or [message]
        first = lines[0].removeprefix("error: ")
        content = {"ename": "TelvaxError", "evalue": first, "traceback": lines}
        if not silent:
            self.send_response(self.iopub_socket, "error", content)
        return {"status": "error", "execution_count": self.execution_count, **content}

    def _command(self, line: str) -> tuple[str, str | None]:
        """A ``:command`` cell: its text, and an error if it was wrong."""
        word, _, rest = line.strip().partition(" ")
        rest = rest.strip()
        if word in (":help", ":h", ":?"):
            return HELP, None
        if word == ":reset":
            self.repl.reset()
            return "(everything forgotten)", None
        if word == ":vars":
            rows = [f"{n}: {t} = {v}" for n, t, v in self.repl.variables()]
            return "\n".join(rows) or "(no variables)", None
        if word == ":defs":
            return "\n".join(self.repl.definitions()) or "(nothing defined)", None
        if word in (":type", ":t"):
            if not rest:
                return "", "error: :type takes an expression, as in `:type 1 + 2`"
            try:
                return self.repl.type_of(rest), None
            except Exception as e:  # TelvaxError
                return "", str(e)
        return "", f"error: no command `{word}`; :help lists them"

    def do_execute(
        self, code, silent, store_history=True, user_expressions=None, allow_stdin=False, *, cell_id=None
    ):
        stripped = code.strip()
        if stripped.startswith(":"):
            text, error = self._command(stripped)
            if error:
                return self._failure(error, silent)
            if not silent:
                self._stream("stdout", text + "\n")
            return self._ok()
        result = self.repl.eval(code)
        if not silent:
            self._stream("stdout", result["output"])
        if result["error"]:
            return self._failure(result["error"], silent)
        value = result["value"]
        if value is not None and not silent:
            data = {"text/plain": value["text"]}
            if value["html"]:
                data["text/html"] = value["html"]
            self.send_response(
                self.iopub_socket,
                "execute_result",
                {"execution_count": self.execution_count, "data": data, "metadata": {}},
            )
        for note in result["notes"]:
            self._stream("stderr", f"note: {note}\n")
        return self._ok()

    def _ok(self) -> dict:
        return {
            "status": "ok",
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }

    # ---- editing help ----------------------------------------------------------------------

    def do_is_complete(self, code):
        if Session.is_complete(code):
            return {"status": "complete"}
        return {"status": "incomplete", "indent": "    "}

    def do_complete(self, code, cursor_pos):
        start, found = self.repl.complete(code, cursor_pos)
        return {
            "status": "ok",
            "matches": [label for label, _ in found],
            "cursor_start": start,
            "cursor_end": cursor_pos,
            "metadata": {
                "_jupyter_types_experimental": [
                    {"text": label, "type": detail or ""} for label, detail in found
                ]
            },
        }

    def do_inspect(self, code, cursor_pos, detail_level=0, omit_sections=()):
        markdown = self.repl.hover(code, cursor_pos)
        if not markdown:
            return {"status": "ok", "found": False, "data": {}, "metadata": {}}
        return {
            "status": "ok",
            "found": True,
            "data": {"text/plain": _fence(markdown), "text/markdown": markdown},
            "metadata": {},
        }


# ---- installing and launching ----------------------------------------------------------------


def kernel_spec() -> dict:
    """The ``kernel.json`` that starts this kernel with the Python that is running now."""
    return {
        "argv": [sys.executable, "-m", "telvax.kernel", "-f", "{connection_file}"],
        "display_name": "Telvax",
        "language": "telvax",
    }


def install(user: bool = True, prefix: str | None = None, name: str = KERNEL_NAME) -> str:
    """Registers the kernel so that Jupyter lists it. Returns where it was put."""
    from jupyter_client.kernelspec import KernelSpecManager

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "kernel.json").write_text(json.dumps(kernel_spec(), indent=2))
        return KernelSpecManager().install_kernel_spec(
            tmp, kernel_name=name, user=user and prefix is None, prefix=prefix
        )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["install"]:
        import argparse

        parser = argparse.ArgumentParser(prog="python -m telvax.kernel install")
        where = parser.add_mutually_exclusive_group()
        where.add_argument("--user", action="store_true", help="for this user (the default)")
        where.add_argument("--sys-prefix", action="store_true", help="in this Python environment")
        where.add_argument("--prefix", help="under this directory")
        parser.add_argument("--name", default=KERNEL_NAME, help="the kernel's name")
        args = parser.parse_args(argv[1:])
        prefix = sys.prefix if args.sys_prefix else args.prefix
        print(install(user=prefix is None, prefix=prefix, name=args.name))
        return
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=TelvaxKernel, argv=argv)


if __name__ == "__main__":
    main()
