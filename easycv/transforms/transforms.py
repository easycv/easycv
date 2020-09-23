from easycv.transforms.noise import Noise
from easycv.transforms.filter import Blur, Sharpness, Sharpen
from easycv.transforms.perspective import Perspective
from easycv.transforms.edges import Gradient, GradientAngle, Canny
from easycv.transforms.color import (
    ColorPick,
    GammaCorrection,
    GrayScale,
    FilterChannels,
    PhotoSketch,
    Negative,
    Cartoon,
    Sepia,
    ColorTransfer,
    Colorize,
    Quantitization,
)
from easycv.transforms.spatial import (
    Resize,
    Rescale,
    Crop,
    Mirror,
    Rotate,
    Translate,
)
from easycv.transforms.selectors import Select
from easycv.transforms.detect import Scan, Eyes, Faces, Smile, Lines, Circles, Detect
from easycv.transforms.draw import Draw
from easycv.transforms.morphological import Erode, Dilate, Morphology

transforms = [
    Blur,
    Canny,
    Circles,
    Cartoon,
    Colorize,
    ColorPick,
    ColorTransfer,
    Crop,
    Eyes,
    Faces,
    Draw,
    Detect,
    Dilate,
    Erode,
    FilterChannels,
    GammaCorrection,
    Gradient,
    GradientAngle,
    GrayScale,
    Mirror,
    Morphology,
    Lines,
    Negative,
    Noise,
    Perspective,
    PhotoSketch,
    Quantitization,
    Rescale,
    Resize,
    Rotate,
    Scan,
    Select,
    Sepia,
    Sharpen,
    Sharpness,
    Smile,
    Translate,
]

transform_names = [transform.__name__ for transform in transforms]


def get_transform(name):
    return transforms[transform_names.index(name)]
