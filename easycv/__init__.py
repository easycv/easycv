# Set version number
__version__ = "0.2.1"

import os
import shutil
import atexit
from pathlib import Path

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
from easycv.camera import Camera
from easycv.video import Video


os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Camera", "Pipeline", "List", "Video"]


@atexit.register
def clear_cache():
    cache_folder = Path(__file__).parent.absolute() / "cache"
    if cache_folder.is_dir():
        shutil.rmtree(str(cache_folder))


clear_cache()
