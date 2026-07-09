from pathlib import Path

from aswaxs_live import xanos_bridge


def test_windows_default_path_is_skipped_on_posix(monkeypatch) -> None:
    monkeypatch.setattr(xanos_bridge, "PLATFORM_NAME", "posix")

    assert not xanos_bridge._candidate_is_compatible(Path(r"C:\Users\jiajtian\XAnoS\XAnoS_Components.py"))
    assert xanos_bridge._candidate_is_compatible(Path("/home/chem_epics/cars6/Data/chemmat/ASWAXS/XAnoS/XAnoS_Components.py"))


def test_default_candidates_include_server_sibling_layout(monkeypatch) -> None:
    project_dir = Path("/home/chem_epics/cars6/data/chemmat/ASWAXS/ASWAXS/ASWAXS_reduction_pipeline_v5")
    monkeypatch.setattr(xanos_bridge, "PROJECT_DIR", project_dir)
    monkeypatch.setattr(xanos_bridge, "PLATFORM_NAME", "posix")
    monkeypatch.delenv("ASWAXS_XANOS_COMPONENTS", raising=False)

    candidates = [str(path).replace("\\", "/") for path in xanos_bridge._default_xanos_candidates()]

    assert "/home/chem_epics/cars6/data/chemmat/ASWAXS/XAnoS/XAnoS_Components.py" in candidates
    assert not any(path.startswith("C:\\") for path in candidates)
