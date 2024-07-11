"""Test versioning."""

from importlib.metadata import version

import nonadditivity


def test_version() -> None:
    """Test that package has version and it is not None."""
    assert hasattr(nonadditivity, "__version__")
    assert nonadditivity.__version__ is not None
    assert nonadditivity.__version__ == version("nonadditivity")
