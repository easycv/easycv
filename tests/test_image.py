from cv import Image, Pipeline
from cv.transforms.color import GrayScale, FilterChannels
from cv.transforms.filter import Blur


def test_image():
    image = Image("tests/images/lenna.png")
    assert image.array is not None


def test_load():
    image = Image("tests/images/lenna.png")
    assert image.loaded
    image = Image("tests/images/lenna.png", lazy=True)
    assert not image.loaded
    image.load()
    assert image.loaded


def test_pending():
    image = Image("tests/images/lenna.png", lazy=True)
    image = image.apply(FilterChannels(channels=[0, 1]))
    image.apply(GrayScale(), in_place=True)
    assert image.pending.num_transforms() == 2


def test_height():
    image = Image("tests/images/lenna.png")
    assert image.height == 512


def test_width():
    image = Image("tests/images/lenna.png")
    assert image.width == 512


def test_compute():
    image = Image("tests/images/lenna.png", lazy=True)
    image = image.apply(GrayScale())
    assert image.pending.num_transforms() == 1
    computed = image.compute()
    assert computed.pending.num_transforms() == 0
    image.compute(in_place=True)
    assert image.pending.num_transforms() == 0


def test_apply():
    image = Image("tests/images/lenna.png")
    pipe = Pipeline([Blur(), GrayScale()])
    image2 = image.apply(Blur()).apply(GrayScale())
    image = image.apply(pipe)
    assert image == image2
