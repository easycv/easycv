# Set version number
__version__ = "0.2.1"

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
from easycv.camera import Camera

import os

os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Camera", "Pipeline", "List"]
