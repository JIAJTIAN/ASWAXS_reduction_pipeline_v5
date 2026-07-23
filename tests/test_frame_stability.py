from pathlib import Path
import h5py
import numpy as np

from aswaxs_live.reduction.frame_qc import (
    FrameSeries,
    FrameStabilityResult,
    analyze_frame_series,
    cormap_p_value,
    discover_frame_source_series,
    discover_stored_frame_stability_results,
    write_frame_stability_shard,
    write_frame_stability_results,
)
from aswaxs_live.reduction.aswaxs_sequence import FrameCurve, ManifestItem, average_groups


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


def test_embedded_reduction_record_takes_priority_over_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "selected_manifest.csv"
    manifest.write_text(
        "sequence_index,energy_index,group_index,frame_index,hdf5_path\n"
        f"1,9,9,1,{tmp_path / 'wrong.h5'}\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(analysis, "w") as handle:
        process = handle.create_group("/entry/process_01_reduction")
        metadata = process.create_group("metadata")
        parameters = process.create_group("parameters")
        process.create_group("data")
        frame_log = process.create_group("frame_filter_log")
        metadata.create_dataset("input_h5_file", data=str(manifest), dtype=string_dtype)
        metadata.create_dataset("input_data_path", data="entry/data/data", dtype=string_dtype)
        parameters.create_dataset("poni_file", data=str(tmp_path / "pil.poni"), dtype=string_dtype)
        parameters.create_dataset("mask_file", data=str(tmp_path / "pil.msk"), dtype=string_dtype)
        parameters.create_dataset("detector", data="Pil300K", dtype=string_dtype)
        parameters.create_dataset("normalization_method", data="monitor:SPDS", dtype=string_dtype)
        parameters.create_dataset("n_q_bins", data=1000)
        parameters.create_dataset("q_unit", data="q_A^-1", dtype=string_dtype)
        frame_log.create_dataset("sequence_index", data=[1, 2])
        frame_log.create_dataset("energy_index", data=[1, 1])
        frame_log.create_dataset("group_index", data=[3, 3])
        frame_log.create_dataset("frame_index", data=[1, 2])
        frame_log.create_dataset(
            "source_file",
            data=np.asarray([str(tmp_path / "raw_1.h5"), str(tmp_path / "raw_2.h5")], dtype=object),
            dtype=string_dtype,
        )

    sources = discover_frame_source_series(analysis)

    assert list(sources) == ["Pil300K | E001 G003"]
    assert [item.path.name for item in sources["Pil300K | E001 G003"].items] == ["raw_1.h5", "raw_2.h5"]


def test_stored_averaging_qc_round_trip(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.h5"
    with h5py.File(analysis, "w") as handle:
        handle.create_group("/entry/process_01_reduction")
    q_shape = np.exp(-np.linspace(0.0, 2.0, 20))
    series = _series(np.asarray([q_shape, q_shape * 1.01]), sigma=0.02)
    result = analyze_frame_series(series)

    class Average:
        energy_index = 1
        group_index = 3
        energy_kev = 12.0
        frame_count = 2
        frame_qc: FrameStabilityResult = result

    write_frame_stability_results(
        analysis,
        "/entry/process_01_reduction",
        "Pil300K",
        [Average()],
    )

    stored = discover_stored_frame_stability_results(analysis)

    assert list(stored) == ["Pil300K | E001 G003"]
    restored = stored["Pil300K | E001 G003"].result
    assert restored.labels == result.labels
    assert restored.recommended.tolist() == result.recommended.tolist()
    np.testing.assert_allclose(restored.intensity_common, result.intensity_common, rtol=1e-6)
    with h5py.File(analysis, "r") as handle:
        assert bool(handle["/entry/process_01_reduction/frame_stability_qc"].attrs["qc_complete"])


def test_group_average_calculates_qc_from_existing_frame_curves(tmp_path: Path) -> None:
    q = np.linspace(0.01, 0.3, 20)
    curves = []
    for frame_index, scale in enumerate((1.0, 1.01), start=1):
        intensity = np.exp(-5.0 * q) * scale
        curves.append(
            FrameCurve(
                item=ManifestItem(frame_index, 1, 2, frame_index, tmp_path / f"raw_{frame_index}.h5"),
                energy_kev=12.0,
                monitor_value=1.0,
                q=q,
                intensity=intensity,
                intensity_error=np.full_like(q, 0.01),
                total_intensity=float(np.sum(intensity)),
                normalized_intensity=intensity,
                normalized_error=np.full_like(q, 0.01),
            )
        )

    averages = average_groups(curves, zmax=5.0)

    assert len(averages) == 1
    assert averages[0].frame_qc is not None
    assert averages[0].frame_qc.frame_index.tolist() == [1, 2]


def test_single_frame_average_skips_stability_qc(tmp_path: Path) -> None:
    q = np.linspace(0.01, 0.3, 20)
    intensity = np.exp(-5.0 * q)
    curve = FrameCurve(
        item=ManifestItem(1, 1, 2, 1, tmp_path / "raw_1.h5"),
        energy_kev=12.0,
        monitor_value=1.0,
        q=q,
        intensity=intensity,
        intensity_error=np.full_like(q, 0.01),
        total_intensity=float(np.sum(intensity)),
        normalized_intensity=intensity,
        normalized_error=np.full_like(q, 0.01),
    )

    average = average_groups([curve], zmax=5.0)[0]

    assert average.frame_qc is None
    assert average.frame_qc_status == "not_applicable_single_frame"


def test_qc_shard_merges_without_result_on_average(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.h5"
    shard = tmp_path / "qc_tmp" / "worker_1.h5"
    with h5py.File(analysis, "w") as handle:
        handle.create_group("/entry/process_01_reduction")
    q_shape = np.exp(-np.linspace(0.0, 2.0, 20))
    result = analyze_frame_series(_series(np.asarray([q_shape, q_shape * 1.01]), sigma=0.02))

    class Average:
        energy_index = 2
        group_index = 4
        energy_kev = 12.1
        frame_count = 2
        frame_qc = result
        frame_qc_status = "complete"
        frame_qc_shard = None
        frame_qc_group = None

    average = Average()
    average.frame_qc_group = write_frame_stability_shard(shard, "Pil300K", average, result)
    average.frame_qc_shard = str(shard)
    average.frame_qc = None

    write_frame_stability_results(analysis, "/entry/process_01_reduction", "Pil300K", [average])

    stored = discover_stored_frame_stability_results(analysis)
    assert stored["Pil300K | E002 G004"].result is not None
    assert not shard.exists()


def test_single_frame_qc_status_round_trip(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.h5"
    with h5py.File(analysis, "w") as handle:
        handle.create_group("/entry/process_01_reduction")

    class Average:
        energy_index = 1
        group_index = 1
        energy_kev = 12.0
        frame_count = 1
        frame_qc = None
        frame_qc_status = "not_applicable_single_frame"
        frame_qc_shard = None
        frame_qc_group = None

    write_frame_stability_results(analysis, "/entry/process_01_reduction", "Pil300K", [Average()])

    stored = discover_stored_frame_stability_results(analysis)["Pil300K | E001 G001"]
    assert stored.result is None
    assert stored.status == "not_applicable_single_frame"
    assert "only one frame" in stored.message
