"""
Note: the caching mechanism and functions are heavily inspired by: https://github.com/allenai/cached_path
"""

import os
from pathlib import Path

from platformdirs import user_cache_dir

CACHE_DIRECTORY: str | Path = os.getenv("TOPICAL_CACHE_ROOT", user_cache_dir("topical", "ai2"))


def get_cache_dir() -> Path:
    """
    Get the global default cache directory.
    """
    return Path(CACHE_DIRECTORY)
