from aswaxs_live.task_queue import TaskSpec, _detector_batch_command, task_from_json


def _task(reduction_mode: str) -> TaskSpec:
    return TaskSpec(
        task_name="mode_test",
        raw_folder="raw",
        output_dir="output",
        num_energies=1,
        num_groups=1,
        num_frames=1,
        pil300k_poni="pil.poni",
        pil300k_mask="pil.msk",
        eig1m_poni="eig.poni",
        eig1m_mask="eig.msk",
        reduction_mode=reduction_mode,
    )


def test_asaxs_mode_predicates() -> None:
    task = _task("asaxs")

    assert task.is_asaxs_mode()
    assert not task.is_saxs_mode()


def test_saxs_mode_predicates() -> None:
    task = _task("saxs")

    assert task.is_saxs_mode()
    assert not task.is_asaxs_mode()


def test_old_task_json_gets_default_monitor_keys() -> None:
    task = task_from_json(
        {
            "task_name": "old",
            "raw_folder": "raw",
            "output_dir": "output",
            "num_energies": 1,
            "num_groups": 1,
            "num_frames": 1,
            "pil300k_poni": "pil.poni",
            "pil300k_mask": "pil.msk",
            "eig1m_poni": "eig.poni",
            "eig1m_mask": "eig.msk",
        }
    )

    assert task.pil300k_monitor_key == "SPDS"
    assert task.eig1m_monitor_key == "WPDS"


def test_detector_batch_command_uses_detector_specific_monitor_key(tmp_path) -> None:
    task = _task("asaxs")
    task.pil300k_monitor_key = "OLD_SPDS"
    task.eig1m_monitor_key = "OLD_WPDS"

    pil_cmd = _detector_batch_command(task, "Pil300K", "pil.poni", "pil.msk", tmp_path / "pil", 1)
    eig_cmd = _detector_batch_command(task, "Eig1M", "eig.poni", "eig.msk", tmp_path / "eig", 1)

    assert pil_cmd[pil_cmd.index("--monitor-key") + 1] == "OLD_SPDS"
    assert eig_cmd[eig_cmd.index("--monitor-key") + 1] == "OLD_WPDS"
