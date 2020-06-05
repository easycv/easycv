# Set version number
__version__ = "0.2.0"

from easycv.image import Image
from easycv.pipeline import Pipeline
from easycv.list import List
import os

os.environ["SESSION_MANAGER"] = ""
__all__ = ["Image", "Pipeline", "List"]
