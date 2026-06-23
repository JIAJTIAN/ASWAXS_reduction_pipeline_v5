from pathlib import Path

import pytest

from aswaxs_live.task_queue import AsaxsPair, TaskSpec, _check_dat_files_writable, _clear_detector_output_records, _ensure_current_pair_xanos_outputs_writable


def _task(output_dir: Path) -> TaskSpec:
    return TaskSpec(
        task_name="demo_task",
        raw_folder=str(output_dir / "raw"),
        output_dir=str(output_dir),
        num_energies=1,
        num_groups=1,
        num_frames=1,
        pil300k_poni="",
        pil300k_mask="",
        eig1m_poni="",
        eig1m_mask="",
        asaxs_pairs=[AsaxsPair("sample A", 1, 1)],
    )


def test_restart_cleanup_keeps_dat_files_and_analysis_results(tmp_path: Path):
    task = _task(tmp_path)
    xanos_sample = tmp_path / "XAnos format" / "sample_A"
    nested = xanos_sample / "manual_analysis"
    nested.mkdir(parents=True)
    dat_file = xanos_sample / "energy_001_sample_A_final.dat"
    nested_dat = nested / "temporary_curve.dat"
    keep_file = nested / "fit_results.csv"
    keep_note = xanos_sample / "notes.txt"
    dat_file.write_text("old generated dat", encoding="utf-8")
    nested_dat.write_text("old nested dat", encoding="utf-8")
    keep_file.write_text("important analysis", encoding="utf-8")
    keep_note.write_text("do not remove", encoding="utf-8")

    _ensure_current_pair_xanos_outputs_writable(tmp_path, task)

    assert xanos_sample.exists()
    assert nested.exists()
    assert dat_file.exists()
    assert nested_dat.exists()
    assert keep_file.exists()
    assert keep_note.exists()


def test_detector_restart_cleanup_keeps_xanos_dat_files(tmp_path: Path):
    detector_dir = tmp_path / "Pil300K"
    xanos_dir = detector_dir / "XAnos format" / "detector_output"
    xanos_dir.mkdir(parents=True)
    dat_file = xanos_dir / "old_curve.dat"
    keep_file = xanos_dir / "manual_fit.csv"
    dat_file.write_text("old generated dat", encoding="utf-8")
    keep_file.write_text("important analysis", encoding="utf-8")

    removed = _clear_detector_output_records(detector_dir)

    assert removed == 0
    assert xanos_dir.exists()
    assert dat_file.exists()
    assert keep_file.exists()


def test_xanos_permission_check_reports_non_writable_dat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "XAnos format" / "sample_A"
    folder.mkdir(parents=True)
    dat_file = folder / "energy_001_sample_A_final.dat"
    dat_file.write_text("old generated dat", encoding="utf-8")
    monkeypatch.setattr("aswaxs_live.task_queue.os.access", lambda path, mode: False)

    with pytest.raises(PermissionError, match="Cannot overwrite existing XAnos .dat"):
        _check_dat_files_writable(folder)
