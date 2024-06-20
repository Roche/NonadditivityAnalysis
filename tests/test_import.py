"""Testing whether nonadditivity can be inported correctly."""
import sys

import nonadditivity  # noqa: F401


def test_nonadditivity_imported() -> None:
    """Sample dev, will always pass so long as import statement worked."""
    assert "nonadditivity" in sys.modules
