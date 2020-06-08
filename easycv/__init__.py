# Set version number
__version__ = "0.2.1"

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
from easycv.video import Video

import os

os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Pipeline", "List", "Video"]
