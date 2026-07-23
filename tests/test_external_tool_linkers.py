from pathlib import Path

from aswaxs_live.tools.linkers import contracts as tool_contracts
from aswaxs_live.tools.linkers import xanos as xanos_linker
from aswaxs_live.tools.linkers import xmodfit as xmodfit_linker

def test_xanos_resolve_from_explicit_path(tmp_path: Path) -> None:
    script = tmp_path / "XAnoS_Components.py"
    script.write_text("class XAnoS_Components: pass\n", encoding="utf-8")

    assert xanos_linker.resolve_xanos_components_path(script) == script.resolve()


def test_xmodfit_resolve_from_explicit_path(tmp_path: Path) -> None:
    script = tmp_path / "xmodfit.py"
    script.write_text("print('xmodfit')\n", encoding="utf-8")

    assert xmodfit_linker.resolve_xmodfit_script(script) == script.resolve()


def test_shared_python_launcher_uses_current_python(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "xmodfit.py"
    script.write_text("print('xmodfit')\n", encoding="utf-8")
    calls = []

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, cwd, env, text):
        calls.append((command, cwd, env, text))
        return FakeProcess()

    monkeypatch.setattr(tool_contracts.subprocess, "Popen", fake_popen)

    process = tool_contracts.launch_python_script(script)

    assert process.poll() is None
    command, cwd, env, text = calls[0]
    assert command[0] == tool_contracts.sys.executable
    assert command[1] == str(script.resolve())
    assert cwd == str(tmp_path.resolve())
    assert str(tmp_path.resolve()) in env["PYTHONPATH"]
    assert text is True
