"""pyFAI bridge helpers used by the calibration GUI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import fabio
import numpy as np
from pyFAI import load as load_poni


def export_image_as_edf(path: str | Path, image: np.ndarray) -> Path:
    """Write an HDF5 detector image to EDF so pyFAI tools can open it."""
    output = Path(path).expanduser().resolve()
    fabio.edfimage.EdfImage(data=np.asarray(image, dtype=np.float32)).write(str(output))
    return output


def export_mask_as_edf(path: str | Path, mask: np.ndarray) -> Path:
    output = Path(path).expanduser().resolve()
    fabio.edfimage.EdfImage(data=np.asarray(mask, dtype=np.uint8)).write(str(output))
    return output


def load_mask_from_edf(path: str | Path) -> np.ndarray:
    with fabio.open(str(Path(path).expanduser().resolve())) as image:
        data = np.asarray(image.data)
    return data.astype(bool)


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    output = Path(path).expanduser().resolve()
    np.save(output, mask.astype(np.uint8))
    return output


def load_mask(path: str | Path) -> np.ndarray:
    mask = np.load(Path(path).expanduser().resolve())
    return mask.astype(bool)


def launch_pyfai_calib2(
    edf_path: str | Path,
    calibrant: str = "AgBh",
    detector_name: str | None = None,
    energy_kev: float | None = None,
    distance_m: float | None = None,
    mask_path: str | Path | None = None,
) -> subprocess.Popen:
    """Launch pyFAI-calib2 with the optional detector geometry hints."""
    edf = Path(edf_path).expanduser().resolve()
    command = ["python", "-m", "pyFAI.app.calib2", str(edf), "--calibrant", calibrant]
    if detector_name:
        command.extend(["--detector", str(detector_name)])
    if energy_kev is not None:
        command.extend(["--energy", str(float(energy_kev))])
    if distance_m is not None:
        command.extend(["--dist", str(float(distance_m))])
    if mask_path is not None:
        command.extend(["--mask", str(Path(mask_path).expanduser().resolve())])
    return subprocess.Popen(command, cwd=edf.parent)


def launch_pyfai_drawmask(edf_path: str | Path) -> subprocess.Popen:
    edf = Path(edf_path).expanduser().resolve()
    command = ["python", "-m", "pyFAI.app.drawmask", str(edf)]
    return subprocess.Popen(command, cwd=edf.parent)


def load_poni_summary(path: str | Path) -> str:
    poni_path = Path(path).expanduser().resolve()
    ai = load_poni(str(poni_path))
    return (
        f"dist={float(ai.dist):.6f} m, "
        f"poni1={float(ai.poni1):.6f} m, "
        f"poni2={float(ai.poni2):.6f} m, "
        f"rot1={float(ai.rot1):.6f}, "
        f"rot2={float(ai.rot2):.6f}, "
        f"rot3={float(ai.rot3):.6f}"
    )
