from pathlib import Path
import h5py
import numpy as np

from aswaxs_live.frame_stability import (
    FrameSeries,
    analyze_frame_series,
    cormap_p_value,
    discover_frame_source_series,
)


def _series(intensity: np.ndarray, sigma: float = 0.01) -> FrameSeries:
    rows, points = intensity.shape
    q = np.linspace(0.01, 0.3, points)
    return FrameSeries(
        q=np.repeat(q.reshape(1, -1), rows, axis=0),
        intensity=intensity,
        sigma=np.full_like(intensity, sigma),
        frame_index=np.arange(1, rows + 1),
        sequence_index=np.arange(1, rows + 1),
        energy_index=np.ones(rows, dtype=int),
        group_index=np.ones(rows, dtype=int),
        energy_kev=np.full(rows, 12.0),
        monitor_value=np.ones(rows),
        source_path=[f"frame_{index}.h5" for index in range(1, rows + 1)],
        existing_status=["accepted"] * rows,
    )


def test_frame_stability_finds_consecutive_drift() -> None:
    q = np.linspace(0.01, 0.3, 200)
    reference = np.exp(-8.0 * q)
    stable = [reference, reference * 1.005, reference * 0.995]
    drifted = [reference * factor for factor in (1.10, 1.12, 1.15)]

    result = analyze_frame_series(_series(np.asarray(stable + drifted), sigma=0.002))

    assert all(label in {"Good", "Acceptable"} for label in result.labels[:3])
    assert result.labels[3:] == ["Bad", "Bad", "Bad"]
    assert result.first_failure_frame == 4
    assert result.damage_onset_frame == 4
    assert result.recommended.tolist() == [True, True, True, False, False, False]


def test_cormap_identical_curves_pass() -> None:
    values = np.linspace(1.0, 2.0, 50)

    p_value, longest = cormap_p_value(values, values.copy())

    assert p_value == 1.0
    assert longest == 0


def test_discovers_frame_series_from_reduction_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "selected_manifest.csv"
    manifest.write_text(
        "sequence_index,energy_index,group_index,frame_index,hdf5_path\n"
        f"1,1,2,1,{tmp_path / 'raw_1.h5'}\n"
        f"2,1,2,2,{tmp_path / 'raw_2.h5'}\n"
        f"3,2,2,1,{tmp_path / 'raw_3.h5'}\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.h5"
    with h5py.File(analysis, "w") as handle:
        process = handle.create_group("/entry/process_01_reduction")
        metadata = process.create_group("metadata")
        parameters = process.create_group("parameters")
        process.create_group("data")
        metadata.create_dataset("input_h5_file", data=str(manifest), dtype=h5py.string_dtype("utf-8"))
        metadata.create_dataset("input_data_path", data="entry/data/data", dtype=h5py.string_dtype("utf-8"))
        parameters.create_dataset("poni_file", data=str(tmp_path / "pil.poni"), dtype=h5py.string_dtype("utf-8"))
        parameters.create_dataset("mask_file", data=str(tmp_path / "pil.msk"), dtype=h5py.string_dtype("utf-8"))
        parameters.create_dataset("detector", data="Pil300K", dtype=h5py.string_dtype("utf-8"))
        parameters.create_dataset("normalization_method", data="monitor:SPDS", dtype=h5py.string_dtype("utf-8"))
        parameters.create_dataset("n_q_bins", data=1000)
        parameters.create_dataset("q_unit", data="q_A^-1", dtype=h5py.string_dtype("utf-8"))

    sources = discover_frame_source_series(analysis)

    assert list(sources) == ["Pil300K | E001 G002", "Pil300K | E002 G002"]
    first = sources["Pil300K | E001 G002"]
    assert len(first.items) == 2
    assert first.monitor_key == "SPDS"
    assert first.npt == 1000
