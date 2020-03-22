import os

from cv.pipeline import Pipeline
from cv.transforms.noise import SaltAndPepper, Impulse, Gaussian


def test_name():
    p = Pipeline([SaltAndPepper(), Impulse(), Gaussian()])
    assert p.name == "pipeline"
    p = Pipeline([SaltAndPepper(), Gaussian()], name="test-pipeline")
    assert p.name == "test-pipeline"


def test_description():
    p = Pipeline([Impulse(), Gaussian(sigma=[])])
    p2 = Pipeline([Impulse(), Gaussian(sigma=[1, 2])])
    assert p.description() == p2.description()
    p3 = Pipeline([Impulse(), Gaussian()])
    p4 = Pipeline([Impulse(prob=0.124), Gaussian()])
    assert p3.description() != str(p4)


def test_num_transforms():
    p = Pipeline([SaltAndPepper(), Impulse(), Gaussian()])
    p2 = Pipeline([SaltAndPepper(), p, Gaussian()])
    assert p2.num_transforms() == 5


def test_transforms():
    p = Pipeline([SaltAndPepper(), Impulse(), Gaussian()])
    assert len(p.transforms()) == p.num_transforms()


def test_save():
    p = Pipeline(
        [SaltAndPepper(prob=0.1512), Impulse(), Gaussian(sigma=5)], name="test"
    )
    p.save()
    assert os.path.exists("test.pipe")
    p2 = Pipeline("test.pipe")
    assert p == p2
    os.remove("test.pipe")
