from aswaxs_live.app.qt_runtime import suppress_glx_warning


def test_suppress_glx_warning_preserves_existing_rules() -> None:
    environment = {"QT_LOGGING_RULES": "qt.foo=true"}

    suppress_glx_warning(environment)

    assert environment["QT_LOGGING_RULES"] == "qt.foo=true;qt.glx=false"


def test_suppress_glx_warning_is_idempotent() -> None:
    environment: dict[str, str] = {}

    suppress_glx_warning(environment)
    suppress_glx_warning(environment)

    assert environment["QT_LOGGING_RULES"] == "qt.glx=false"
