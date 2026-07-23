from pathlib import Path


def test_task_queue_has_single_canonical_module() -> None:
    import aswaxs_live.workflows.queue as imported
    from aswaxs_live.workflows import queue as canonical

    assert imported is canonical
    assert imported.TaskSpec is canonical.TaskSpec


def test_dashboard_has_single_canonical_module() -> None:
    import aswaxs_live.app.dashboard as imported
    from aswaxs_live.app import dashboard as canonical

    assert imported is canonical


def test_task_model_is_shared_with_workflow() -> None:
    from aswaxs_live.workflows.task import TaskSpec
    from aswaxs_live.workflows.queue import TaskSpec as WorkflowTaskSpec

    assert TaskSpec is WorkflowTaskSpec


def test_project_path_is_independent_of_module_depth() -> None:
    from aswaxs_live.paths import PROJECT_DIR
    from aswaxs_live.app.dashboard import PROJECT_DIR as DashboardProjectDir
    from aswaxs_live.workflows.queue import PROJECT_DIR as QueueProjectDir

    assert PROJECT_DIR == DashboardProjectDir == QueueProjectDir
    assert (Path(PROJECT_DIR) / "pyproject.toml").is_file()
