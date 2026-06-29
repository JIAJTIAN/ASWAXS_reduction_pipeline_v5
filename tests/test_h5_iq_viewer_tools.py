from pathlib import Path

import h5py
import numpy as np

from aswaxs_live.h5_tools import H5CurveRecord, _curve_matches_source, _fixed_background_sample_indices, _safe_filename, _subtract_background_curve, _write_curve_export, _xanos_dat_path, discover_iq_curves


def _record(label: str, group_path: str, h5_path: str = "") -> H5CurveRecord:
    return H5CurveRecord(
        label=label,
        h5_path=h5_path,
        group_path=group_path,
        q_path=f"{group_path}/q",
        i_path=f"{group_path}/I",
        y_name="I",
        sigma_path=f"{group_path}/sigma_I",
        row=None,
    )


def test_curve_source_filter_splits_detector_and_combined_curves() -> None:
    saxs = _record("Pil300K group average", "/entry/process_01_reduction/data/group_04", "run/Pil300K/sample_analysis.h5")
    waxs = _record("Eig1M group average", "/entry/process_01_reduction/data/group_04", "run/Eig1M/sample_analysis.h5")
    combined = _record("Stitched / sample | I", "/entry/stitched_averages/curves/sample")

    assert _curve_matches_source(saxs, "saxs")
    assert not _curve_matches_source(saxs, "waxs")
    assert _curve_matches_source(waxs, "waxs")
    assert not _curve_matches_source(waxs, "saxs")
    assert _curve_matches_source(combined, "combined")
    assert not _curve_matches_source(combined, "saxs")
    assert _curve_matches_source(combined, "all")


def test_write_curve_export_carries_header_and_sigma(tmp_path: Path) -> None:
    h5_path = tmp_path / "sample_analysis.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("/entry/stitched_averages/curves/sample")
        group.create_dataset("I", data=np.array([10.0, 9.0]))
        group.attrs["energy_kev"] = 9.3
    record = _record("Stitched / sample | I", "/entry/stitched_averages/curves/sample", str(h5_path))
    target = tmp_path / (_safe_filename(record.label) + ".dat")

    _write_curve_export(
        target,
        np.array([0.1, 0.2]),
        np.array([10.0, 9.0]),
        np.array([0.5, 0.4]),
        label=record.label,
        record=record,
        background_label="solvent",
        background_factor=0.98,
    )

    text = target.read_text(encoding="utf-8")
    assert "#Reduced per-energy I-q curve exported from analysis HDF5" in text
    assert "#Energy=9.300000000" in text
    assert "#CF=1" in text
    assert '"background_factor": 0.98' in text
    assert '"background_label": "solvent"' in text
    assert "#col_names=['Q (inv Angs)','Int','Int_err']" in text
    assert "#columns=q I_final I_final_err" in text
    np.testing.assert_allclose(np.loadtxt(target), [[0.1, 10.0, 0.5], [0.2, 9.0, 0.4]])


def test_xanos_dat_path_replaces_non_dat_suffix() -> None:
    assert _xanos_dat_path(Path("curve.csv")) == Path("curve.dat")
    assert _xanos_dat_path(Path("curve")) == Path("curve.dat")


def test_discover_iq_curves_uses_sigma_as_error_not_curve(tmp_path: Path) -> None:
    h5_path = tmp_path / "sample_analysis.h5"
    q = np.linspace(0.01, 0.2, 20)
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("/entry/stitched_averages/curves/sample")
        group.create_dataset("q", data=q)
        group.create_dataset("I", data=np.exp(-q))
        group.create_dataset("sigma_I", data=np.full_like(q, 0.01))

    with h5py.File(h5_path, "r") as handle:
        curves = discover_iq_curves(handle, source_path=h5_path)

    assert len(curves) == 1
    assert curves[0].i_path.endswith("/I")
    assert curves[0].sigma_path.endswith("/sigma_I")


def test_pair_subtraction_interpolates_background_and_propagates_error() -> None:
    q = np.array([1.0, 2.0, 3.0])
    intensity = np.array([10.0, 20.0, 30.0])
    sigma = np.array([1.0, 1.0, 1.0])
    background_q = np.array([1.0, 3.0])
    background_i = np.array([2.0, 6.0])
    background_sigma = np.array([0.5, 1.5])

    corrected_q, corrected_i, corrected_sigma = _subtract_background_curve(
        q,
        intensity,
        sigma,
        background_q,
        background_i,
        background_sigma,
        factor=0.5,
    )

    np.testing.assert_allclose(corrected_q, q)
    np.testing.assert_allclose(corrected_i, [9.0, 18.0, 27.0])
    np.testing.assert_allclose(corrected_sigma, np.sqrt([1.0**2 + 0.25**2, 1.0**2 + 0.5**2, 1.0**2 + 0.75**2]))


def test_fixed_background_builds_multiple_unique_sample_pairs() -> None:
    assert _fixed_background_sample_indices([1, 2, 4, 2, 7], background_index=4) == [1, 2, 7]
