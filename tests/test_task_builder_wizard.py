import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
from PyQt5 import QtWidgets

from aswaxs_live import dashboard
from aswaxs_live.task_queue import TaskSpec


def test_task_builder_uses_guided_pages(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "BUILDER_SETTINGS_PATH", tmp_path / "builder.json")
    monkeypatch.setattr(dashboard, "DEFAULT_QUEUE_PATH", tmp_path / "queue.json")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = dashboard.DashboardWindow()
    window.show()
    window.tabs.setCurrentIndex(0)
    app.processEvents()

    assert window.tabs.tabText(0) == "Task Builder"
    assert window.tabs.tabText(1) == "Dashboard"
    assert window.builder_step_titles[0] == "Raw Data"
    assert window.builder_stack.count() == 5
    assert window.builder_stack.currentIndex() == 0
    assert not window.builder_back_button.isEnabled()

    sample_folder = tmp_path / "example_sample"
    window.raw_folder_edit.setText(str(sample_folder))
    window.builder_next_button.click()
    assert window.task_name_edit.text() == "example_sample"
    assert window.builder_stack.currentIndex() == 1

    for expected_page in range(2, 5):
        window.builder_next_button.click()
        app.processEvents()
        assert window.builder_stack.currentIndex() == expected_page

    assert window.builder_next_button.isHidden()
    assert not window.add_task_button.isHidden()
    assert window.update_task_button.isHidden()

    window.editing_index = 0
    window._set_builder_step(4)
    assert window.add_task_button.isHidden()
    assert not window.update_task_button.isHidden()
    labels = {label.text() for label in window.findChildren(QtWidgets.QLabel)}
    assert "Sample thickness" in labels
    assert "Capillary thickness" not in labels
    calibration_labels = {label.text() for label in window.builder_stack.widget(3).findChildren(QtWidgets.QLabel)}
    sample_labels = {label.text() for label in window.builder_stack.widget(4).findChildren(QtWidgets.QLabel)}
    assert "GC group" not in calibration_labels
    assert "Air group" not in calibration_labels
    assert "Empty group" not in calibration_labels
    assert "GC group" in sample_labels
    assert "Air group" in sample_labels
    assert "Empty group" in sample_labels
    window.close()


def test_toolbar_does_not_bypass_wizard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "BUILDER_SETTINGS_PATH", tmp_path / "builder.json")
    monkeypatch.setattr(dashboard, "DEFAULT_QUEUE_PATH", tmp_path / "queue.json")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = dashboard.DashboardWindow()

    queue_toolbar = next(toolbar for toolbar in window.findChildren(QtWidgets.QToolBar) if toolbar.windowTitle() == "Queue")
    assert window.add_to_queue_action not in queue_toolbar.actions()
    window.close()


def test_task_builder_step_buttons_jump_directly(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "BUILDER_SETTINGS_PATH", tmp_path / "builder.json")
    monkeypatch.setattr(dashboard, "DEFAULT_QUEUE_PATH", tmp_path / "queue.json")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = dashboard.DashboardWindow()
    sample_folder = tmp_path / "direct_jump_sample"
    window.raw_folder_edit.setText(str(sample_folder))

    calibration_button = next(
        button for button in window.builder_step_buttons if button.text() == "4. Calibration"
    )
    calibration_button.click()
    app.processEvents()

    assert window.builder_stack.currentIndex() == 3
    assert window.task_name_edit.text() == "direct_jump_sample"
    assert calibration_button.property("current") is True
    assert window.builder_step_buttons[0].property("current") is False
    window.builder_step_buttons[1].click()
    app.processEvents()
    assert window.builder_stack.currentIndex() == 1
    window.close()


def test_needs_attention_tooltip_shows_validation_reasons(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "BUILDER_SETTINGS_PATH", tmp_path / "builder.json")
    monkeypatch.setattr(dashboard, "DEFAULT_QUEUE_PATH", tmp_path / "queue.json")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = dashboard.DashboardWindow()
    task = TaskSpec(
        task_name="bad_task",
        raw_folder=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        num_energies=1,
        num_groups=1,
        num_frames=1,
        pil300k_poni="",
        pil300k_mask="",
        eig1m_poni="",
        eig1m_mask="",
        status="Needs Attention",
        message="Pil300K: no files; Pil300K PONI missing; Pil300K mask missing",
    )

    tooltip = window._task_tooltip(task)

    assert "Status: Needs Attention" in tooltip
    assert "Needs attention:" in tooltip
    assert "- Pil300K: no files" in tooltip
    assert "- Pil300K PONI missing" in tooltip
    assert "Task information:" in tooltip
    window.close()


def test_calibration_step_auto_scans_monitor_pvs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "BUILDER_SETTINGS_PATH", tmp_path / "builder.json")
    monkeypatch.setattr(dashboard, "DEFAULT_QUEUE_PATH", tmp_path / "queue.json")
    raw = tmp_path / "sample"
    pil = raw / "Pil300K"
    eig = raw / "Eig1M"
    pil.mkdir(parents=True)
    eig.mkdir(parents=True)
    with h5py.File(pil / "pil_0001.h5", "w") as handle:
        handle.create_dataset("/entry/instrument/NDAttributes/OLD_SPDS", data=1.0)
    with h5py.File(eig / "eig_0001.h5", "w") as handle:
        handle.create_dataset("/entry/instrument/NDAttributes/OLD_WPDS", data=1.0)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = dashboard.DashboardWindow()
    window.raw_folder_edit.setText(str(raw))
    for _step in range(3):
        window.builder_next_button.click()
        app.processEvents()

    assert window.builder_stack.currentIndex() == 3
    assert window.pil_monitor_combo.findText("OLD_SPDS") >= 0
    assert window.eig_monitor_combo.findText("OLD_WPDS") >= 0
    window.close()
