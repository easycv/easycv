# Set version number
__version__ = "0.2.1"

import shutil
from pathlib import Path
import atexit

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
from easycv.video import Video
from easycv.dataset import Dataset


import os

os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Pipeline", "List", "Video", "Dataset"]


@atexit.register
def clear_cache():
    cache_folder = Path(__file__).parent.absolute() / "cache"
    if cache_folder.is_dir():
        shutil.rmtree(str(cache_folder))


clear_cache()
