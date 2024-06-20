"""Custom errors to be thrown when running mmpdb."""


class FragmentationError(Exception):
    """Thrown when mmpdb fragment fails."""


class IndexingError(Exception):
    """Thrown when mmpdb index fails."""
