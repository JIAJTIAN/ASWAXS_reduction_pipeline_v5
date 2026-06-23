from pathlib import Path

import numpy as np

from aswaxs_live.stitcher import ReductionRows, scale_high_q_to_low_q, stitch_one_row


def _power_curve(q, amplitude=1.0, exponent=-2.0, sigma_fraction=0.05):
    intensity = amplitude * np.power(q, exponent)
    sigma = sigma_fraction * intensity
    return np.column_stack([q, intensity, sigma])


def _rows(detector, q, amplitude=1.0, exponent=-2.0):
    curve = _power_curve(q, amplitude=amplitude, exponent=exponent)
    return ReductionRows(
        detector=detector,
        path=Path(f"{detector}.h5"),
        q=q,
        intensity=curve[:, 1][None, :],
        sigma=curve[:, 2][None, :],
        energy=np.array([10.0]),
        energy_index=np.array([1]),
        group_index=np.array([1]),
        mtime_ns=1,
        size=1,
    )


def test_overlap_scaling_uses_median_ratio():
    low_q = _power_curve(np.linspace(0.05, 0.30, 200), amplitude=10.0)
    high_q = _power_curve(np.linspace(0.20, 0.60, 200), amplitude=2.0)

    scale, q_min, q_max, n_overlap = scale_high_q_to_low_q(low_q, high_q, overlap_q_max=0.28)

    assert np.isclose(scale, 5.0, rtol=1e-6)
    assert q_min >= 0.20
    assert q_max <= 0.28
    assert n_overlap >= 3


def test_gap_scaling_uses_edge_extrapolation():
    low_q = _power_curve(np.linspace(0.05, 0.20, 120), amplitude=10.0)
    high_q = _power_curve(np.linspace(0.30, 0.70, 120), amplitude=2.0)

    scale, q_min, q_max, n_overlap = scale_high_q_to_low_q(low_q, high_q, overlap_q_max=0.28)

    assert np.isclose(scale, 5.0, rtol=1e-6)
    assert np.isclose(q_min, 0.20)
    assert np.isclose(q_max, 0.30)
    assert n_overlap == 0


def test_stitch_one_row_keeps_both_detectors_across_q_gap():
    low_rows = _rows("Pil300K", np.linspace(0.05, 0.20, 120), amplitude=10.0)
    high_rows = _rows("Eig1M", np.linspace(0.30, 0.70, 120), amplitude=2.0)

    stitched, source_detector, scale, q_min, q_max, join_q, n_overlap, low_count, high_count = stitch_one_row(
        low_rows,
        high_rows,
        row=0,
        overlap_q_max=0.28,
    )

    assert np.isclose(scale, 5.0, rtol=1e-6)
    assert n_overlap == 0
    assert q_min < join_q < q_max
    assert low_count == 120
    assert high_count == 120
    assert set(source_detector.tolist()) == {"Pil300K", "Eig1M"}
    assert np.all(np.diff(stitched[:, 0]) > 0)
