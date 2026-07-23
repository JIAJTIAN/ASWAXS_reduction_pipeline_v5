from pathlib import Path

import h5py
import numpy as np
import pytest
from pyFAI import AzimuthalIntegrator

from aswaxs_live.reduction.analysis_h5 import write_reduction_to_analysis_h5
from aswaxs_live.reduction.pipeline import poni_geometry_metadata


def test_poni_geometry_metadata_contains_beam_center_and_distance(tmp_path: Path) -> None:
    poni = tmp_path / "geometry.poni"
    integrator = AzimuthalIntegrator(
        dist=0.245,
        poni1=0.0034,
        poni2=0.0046,
        pixel1=0.0001,
        pixel2=0.0002,
    )
    integrator.save(str(poni))

    metadata = poni_geometry_metadata(poni)

    assert metadata["sample_detector_distance_m"] == pytest.approx(0.245)
    assert metadata["sample_detector_distance_mm"] == pytest.approx(245.0)
    assert metadata["beam_center_x_px"] == pytest.approx(23.0)
    assert metadata["beam_center_y_px"] == pytest.approx(34.0)
    assert metadata["geometry_source"] == "PONI"


def test_reduction_h5_records_geometry_in_process_metadata(tmp_path: Path) -> None:
    raw = tmp_path / "raw.h5"
    analysis = tmp_path / "analysis.h5"
    with h5py.File(raw, "w") as handle:
        handle.create_dataset("entry/data/data", data=np.ones((4, 4), dtype=np.float32))

    geometry = {
        "beam_center_x_px": 123.5,
        "beam_center_y_px": 456.5,
        "sample_detector_distance_m": 0.245,
        "sample_detector_distance_mm": 245.0,
        "geometry_source": "PONI",
    }
    write_reduction_to_analysis_h5(
        analysis,
        raw,
        np.linspace(0.01, 0.1, 5),
        np.ones((1, 5)),
        np.full((1, 5), 0.1),
        {
            "input_data_path": "entry/data/data",
            "n_total_frames": 1,
            "n_accepted_frames": 1,
            "n_rejected_frames": 0,
            **geometry,
        },
        {"integration_method": "pyFAI.integrate1d"},
    )

    with h5py.File(analysis, "r") as handle:
        metadata = handle["/entry/process_01_reduction/metadata"]
        assert metadata["beam_center_x_px"][()] == pytest.approx(123.5)
        assert metadata["beam_center_y_px"][()] == pytest.approx(456.5)
        assert metadata["sample_detector_distance_m"][()] == pytest.approx(0.245)
        assert metadata["sample_detector_distance_mm"][()] == pytest.approx(245.0)
