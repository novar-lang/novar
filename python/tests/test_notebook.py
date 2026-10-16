"""The interactive session, the Jupyter kernel (over real sockets) and the ``%%telvax`` magic."""

import os

import pytest

import telvax


# ---- the session ------------------------------------------------------------------------------


def test_a_session_keeps_definitions_and_variables():
    s = telvax.Session()
    assert s.eval("let x = 5")["defined"] == ["x"]
    assert s.eval("x * 2")["value"]["text"] == "10"
    s.eval("fn double(n: Int) -> Int:\n    n * 2")
    assert s.eval("double(x)")["value"] == {"text": "10", "type": "Int", "html": None}
    assert s.definitions() == ["fn double"]
    assert s.variables() == [("x", "Int", "5")]
    s.reset()
    assert s.definitions() == [] and s.variables() == []


def test_output_errors_and_types():
    s = telvax.Session()
    assert s.eval('print("hi")')["output"] == "hi\n"
    result = s.eval('1 + "a"')
    assert result["error"].startswith("error: mismatched operand types")
    s.eval("let xs = [1, 2]")
    assert s.eval("xs")["value"]["type"] == "List[Int]"
    assert s.type_of("1.5") == "Float"
    with pytest.raises(telvax.TelvaxError):
        s.type_of("1 +")
    assert not telvax.Session.is_complete("fn f():")
    assert telvax.Session.is_complete("1 + 2")


def test_a_session_can_be_limited():
    s = telvax.Session(allow="none", step_limit=10_000)
    assert "needs the `fs.read` capability" in s.eval('std.fs.read("/etc/hostname")')["error"]
    assert s.eval("while True:\n    pass")["error"]
    assert s.eval("1 + 1")["value"]["text"] == "2"
    with pytest.raises(telvax.TelvaxError):
        telvax.Session(allow="nonsense")


def test_a_tensor_comes_with_html():
    s = telvax.Session()
    s.eval("let t: Tensor[f32, 2, 3] = std.tensor.zeros()")
    html = s.eval("t")["value"]["html"]
    assert "telvax-tensor" in html and "[2, 3]" in html


def test_positions_are_characters_not_bytes():
    s = telvax.Session()
    s.eval("let origem = 1")
    # `é` takes two bytes, one character: the answers count characters.
    code = 'print("é") + orig'
    start, found = s.complete(code, len(code))
    assert code[start:] == "orig"
    assert any(label == "origem" for label, _ in found)
    code = 'print("é")\norigem'
    assert "let origem: Int" in s.hover(code, code.index("origem") + 2)


# ---- the Jupyter kernel -------------------------------------------------------------------------

jupyter_client = pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    from jupyter_client.manager import start_new_kernel

    from telvax.kernel import install

    prefix = tmp_path_factory.mktemp("jupyter")
    install(prefix=str(prefix))
    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(prefix / "share" / "jupyter")
    manager, client = start_new_kernel(kernel_name="telvax", startup_timeout=60)
    yield client
    client.stop_channels()
    manager.shutdown_kernel(now=True)
    if old is None:
        os.environ.pop("JUPYTER_PATH", None)
    else:
        os.environ["JUPYTER_PATH"] = old


def run(client, code):
    """Runs a cell and returns its reply and what it published, by kind."""
    msg_id = client.execute(code)
    published = {"stream": [], "execute_result": [], "error": []}
    while True:
        msg = client.get_iopub_msg(timeout=60)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind = msg["msg_type"]
        if kind == "status" and msg["content"]["execution_state"] == "idle":
            break
        if kind in published:
            published[kind].append(msg["content"])
    reply = client.get_shell_msg(timeout=60)["content"]
    return reply, published


def test_the_kernel_describes_itself(kernel):
    kernel.kernel_info()
    reply = kernel.get_shell_msg(timeout=30)["content"]
    assert reply["language_info"]["name"] == "telvax"
    assert reply["language_info"]["file_extension"] == ".tvx"
    assert reply["implementation"] == "Telvax"


def test_cells_share_definitions_and_variables(kernel):
    reply, out = run(kernel, "let x = 5")
    assert reply["status"] == "ok" and not out["execute_result"]
    reply, out = run(kernel, "fn double(n: Int) -> Int:\n    n * 2\n")
    assert reply["status"] == "ok"
    reply, out = run(kernel, "double(x)")
    assert out["execute_result"][0]["data"]["text/plain"] == "10"
    reply, out = run(kernel, 'print("hello")\nx + 1')
    assert out["stream"][0] == {"name": "stdout", "text": "hello\n"}
    assert out["execute_result"][0]["data"]["text/plain"] == "6"


def test_an_error_is_reported_and_the_kernel_goes_on(kernel):
    reply, out = run(kernel, 'let y = 1\ny + "a"')
    assert reply["status"] == "error" and reply["ename"] == "TelvaxError"
    assert "mismatched operand types" in reply["evalue"]
    assert out["error"][0]["traceback"][0].startswith("error: mismatched")
    reply, out = run(kernel, "y")
    assert reply["status"] == "error", "the cell that failed defined nothing"
    reply, out = run(kernel, "1 + 1")
    assert out["execute_result"][0]["data"]["text/plain"] == "2"


def test_a_tensor_is_shown_as_html(kernel):
    run(kernel, "let t: Tensor[f32, 2, 2] = std.tensor.zeros()")
    reply, out = run(kernel, "t")
    data = out["execute_result"][0]["data"]
    assert "telvax-tensor" in data["text/html"] and "text/plain" in data


def test_colon_commands_work_in_cells(kernel):
    run(kernel, "let origin = [1, 2]")
    reply, out = run(kernel, ":type origin")
    assert out["stream"][0]["text"] == "List[Int]\n"
    reply, out = run(kernel, ":vars")
    assert "origin: List[Int] = [1, 2]" in out["stream"][0]["text"]
    reply, out = run(kernel, ":nope")
    assert reply["status"] == "error"


def test_the_notebook_can_ask_whether_a_cell_is_finished(kernel):
    kernel.is_complete("fn f() -> Int:")
    reply = kernel.get_shell_msg(timeout=30)["content"]
    assert reply == {"status": "incomplete", "indent": "    "}
    kernel.is_complete("1 + 2")
    assert kernel.get_shell_msg(timeout=30)["content"]["status"] == "complete"


def test_tab_and_shift_tab(kernel):
    run(kernel, "let origem = 3")
    kernel.complete("print(orig", 10)
    reply = kernel.get_shell_msg(timeout=30)["content"]
    assert "origem" in reply["matches"] and reply["cursor_start"] == 6
    kernel.inspect("origem", 3)
    reply = kernel.get_shell_msg(timeout=30)["content"]
    assert reply["found"] and "let origem: Int" in reply["data"]["text/plain"]
    kernel.inspect("zzz", 1)
    assert kernel.get_shell_msg(timeout=30)["content"]["found"] is False


def test_the_installed_kernel_spec_names_this_python():
    from telvax.kernel import kernel_spec

    spec = kernel_spec()
    assert spec["language"] == "telvax" and spec["argv"][1:3] == ["-m", "telvax.kernel"]


# ---- %%telvax ---------------------------------------------------------------------------------


def test_the_cell_magic_runs_telvax_in_a_python_notebook():
    pytest.importorskip("IPython")
    from IPython.core.error import UsageError
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "telvax")
    with capture_output() as captured:
        shell.run_cell_magic("telvax", "", "let x = 20\nprint(\"seen\")\nx + 1")
    assert "seen" in captured.stdout
    assert captured.outputs[0].data["text/plain"] == "21"
    with capture_output() as captured:
        shell.run_cell_magic("telvax", "", "x * 2")
    assert captured.outputs[0].data["text/plain"] == "40"
    with pytest.raises(UsageError, match="mismatched operand types"):
        shell.run_cell_magic("telvax", "", 'x + "a"')
    shell.run_line_magic("telvax_reset", "")
    with pytest.raises(UsageError):
        shell.run_cell_magic("telvax", "", "x")
