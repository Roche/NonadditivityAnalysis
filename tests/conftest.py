"""Configuring test environment for nonadditivity analysis package."""

from collections.abc import Sequence

import pytest

pytest_plugins = [
    "tests._fixtures.classification_fixtures",
    "tests._fixtures.commandline_fixtures",
    "tests._fixtures.dataframes_fixtures",
    "tests._fixtures.path_fixtures",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Helper function to skip slow tests.

    use argument --run-slow to also run the
    slow tests with pytest.

    Args:
        parser (pytest.Parser): pytest argument parser
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Add 'slow' to available markers for pytest.mark.

    Args:
        config (pytest.Coinfig): pytest config item
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: Sequence[pytest.Item],
) -> None:
    """Connecgt --run-slow option to mark.slow decorateor.

    Args:
        config (pytest.Config): pytest config item
        items (Sequence[pytest.Item]): pytest items
    """
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
