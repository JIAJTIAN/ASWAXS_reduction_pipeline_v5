from aswaxs_live.workflows.queue import _SmoothedFrameRate


def test_smoothed_frame_rate_is_stable_for_constant_throughput() -> None:
    estimator = _SmoothedFrameRate(0.0)

    first = estimator.update(100, now=10.0)
    second = estimator.update(200, now=20.0)

    assert first == 10.0
    assert second == 10.0
    assert estimator.update(200, now=25.0) == second


def test_smoothed_frame_rate_responds_gradually_to_slowdown() -> None:
    estimator = _SmoothedFrameRate(0.0, window_seconds=30.0, alpha=0.25)
    estimator.update(100, now=10.0)
    fast_rate = estimator.update(200, now=20.0)

    slowed_once = estimator.update(210, now=30.0)
    slowed_twice = estimator.update(220, now=40.0)

    assert fast_rate == 10.0
    assert 1.0 < slowed_twice < slowed_once < fast_rate


def test_smoothed_frame_rate_resets_if_completed_count_moves_backward() -> None:
    estimator = _SmoothedFrameRate(0.0)
    estimator.update(100, now=10.0)

    assert estimator.update(20, now=20.0) == 0.0
    assert estimator.update(40, now=22.0) == 10.0
