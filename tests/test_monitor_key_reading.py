import h5py

from aswaxs_live.core.reduce_aswaxs_sequence import read_ndattr_scalar_from_handle


def test_read_monitor_key_by_ndattribute_name(tmp_path) -> None:
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/instrument/NDAttributes/OLD_SPDS", data=12.5)

    with h5py.File(path, "r") as handle:
        assert read_ndattr_scalar_from_handle(handle, "OLD_SPDS") == 12.5


def test_read_monitor_key_by_direct_hdf5_path(tmp_path) -> None:
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/entry/metadata/beam_monitor", data=7.25)

    with h5py.File(path, "r") as handle:
        assert read_ndattr_scalar_from_handle(handle, "/entry/metadata/beam_monitor") == 7.25
