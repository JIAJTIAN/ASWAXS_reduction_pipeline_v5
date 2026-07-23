from pathlib import Path

import h5py
import numpy as np

from aswaxs_live.app.dashboard import recovered_result_task
from aswaxs_live.workflows.queue import preflight_task


def test_recover_completed_asaxs_result_for_preview(tmp_path: Path) -> None:
    path = tmp_path / "sample_analysis.h5"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        outputs = entry.create_group("asaxs_outputs")
        data = outputs.create_group("sample").create_group("corrected_I_q_E")
        data.create_dataset("q", data=np.tile(np.linspace(0.01, 0.2, 10), (2, 1)))
        data.create_dataset("I", data=np.ones((2, 10)))
        process = entry.create_group("process_01_reduction")
        reduction_data = process.create_group("data")
        reduction_data.create_dataset("energy_index", data=[1, 1, 2, 2])
        reduction_data.create_dataset("group_index", data=[1, 2, 1, 2])
        frame_log = process.create_group("frame_filter_log")
        frame_log.create_dataset("frame_index", data=[1, 2, 3, 1, 2, 3])

    task = recovered_result_task(path)

    assert task.task_name == "sample"
    assert task.status == "Done"
    assert task.reduction_mode == "asaxs"
    assert task.num_energies == 2
    assert task.num_groups == 2
    assert task.num_frames == 3
    assert Path(task.analysis_h5_path) == path.resolve()
    assert preflight_task(task) == (
        False,
        "Recovered analysis HDF5 records are preview-only; create or edit a task to run a new reduction",
    )
