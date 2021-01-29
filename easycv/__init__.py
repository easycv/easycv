# Set version number
__version__ = "0.3.0"

import shutil
from pathlib import Path
import atexit

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
from easycv.video import Video

import os

os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Pipeline", "List", "Video"]


@atexit.register
def clear_cache():
    cache_folder = Path(__file__).parent.absolute() / "cache"
    if cache_folder.is_dir():
        shutil.rmtree(str(cache_folder))


clear_cache()
