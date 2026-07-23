from pathlib import Path

import h5py
import numpy as np

from aswaxs_live.reduction.xanos_export import export_analysis_h5_to_xanos_format


def test_saxs_xanos_export_uses_requested_output_name(tmp_path: Path):
    analysis_h5 = tmp_path / "task_name_analysis.h5"
    with h5py.File(analysis_h5, "w") as handle:
        data = handle.create_group("/entry/process_01_reduction/data")
        data.create_dataset("q", data=np.linspace(0.01, 0.1, 5))
        data.create_dataset("I", data=np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        data.create_dataset("sigma_I", data=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]]))
        data.create_dataset("energy", data=np.asarray([12.0]))

    written = export_analysis_h5_to_xanos_format(analysis_h5, saxs_output_name="SAXS_sample_A")

    dat_files = [path for path in written if path.suffix == ".dat"]
    assert len(dat_files) == 1
    assert dat_files[0].parent == tmp_path / "XAnos format" / "SAXS_sample_A"
    assert dat_files[0].name == "SAXS_sample_A_final.dat"
    assert "output_name\": \"SAXS_sample_A" in dat_files[0].read_text(encoding="utf-8")


def test_force_saxs_ignores_stale_final_branch(tmp_path: Path):
    analysis_h5 = tmp_path / "task_name_analysis.h5"
    q = np.linspace(0.01, 0.1, 5)
    with h5py.File(analysis_h5, "w") as handle:
        saxs = handle.create_group("/entry/process_01_reduction/data")
        saxs.create_dataset("q", data=q)
        saxs.create_dataset("I", data=np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0]]))
        saxs.create_dataset("sigma_I", data=np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        saxs.create_dataset("energy", data=np.asarray([12.0]))

        stale = handle.create_group("/entry/final/corrected_I_q_E")
        stale.create_dataset("q", data=q)
        stale.create_dataset("I", data=np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0]]))
        stale.create_dataset("sigma_I", data=np.asarray([[0.1, 0.1, 0.1, 0.1, 0.1]]))
        stale.create_dataset("energy", data=np.asarray([9.0]))

    written = export_analysis_h5_to_xanos_format(analysis_h5, saxs_output_name="SAXS_real_name", force_saxs=True)

    dat_files = [path for path in written if path.suffix == ".dat"]
    assert len(dat_files) == 1
    assert dat_files[0].parent == tmp_path / "XAnos format" / "SAXS_real_name"
    assert dat_files[0].name == "SAXS_real_name_final.dat"
    data = np.loadtxt(dat_files[0])
    assert np.allclose(data[:, 1], [10.0, 20.0, 30.0, 40.0, 50.0])


def test_asaxs_xanos_export_keeps_energy_prefix(tmp_path: Path):
    analysis_h5 = tmp_path / "asaxs_task_analysis.h5"
    q = np.linspace(0.01, 0.1, 5)
    with h5py.File(analysis_h5, "w") as handle:
        data = handle.create_group("/entry/asaxs_outputs/Sample_A/corrected_I_q_E")
        data.create_dataset("q", data=q)
        data.create_dataset("I", data=np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        data.create_dataset("sigma_I", data=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]]))
        data.create_dataset("energy", data=np.asarray([12.0]))

    written = export_analysis_h5_to_xanos_format(analysis_h5)

    dat_files = [path for path in written if path.suffix == ".dat"]
    assert len(dat_files) == 1
    assert dat_files[0].parent == tmp_path / "XAnos format" / "Sample_A"
    assert dat_files[0].name == "energy_001_Sample_A_final.dat"
