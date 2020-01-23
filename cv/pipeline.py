import os
import pickle

from cv.transforms.base import Transform
from cv.errors.io import InvalidPipelineInputSource


class Pipeline(object):
    def __init__(self, source, name=None):
        if isinstance(source, list) and all([isinstance(x, Transform) or isinstance(x, Pipeline) for x in source]):
            self._name = name if name else 'pipeline'
            self._transforms = source
        elif isinstance(source, str) and os.path.isfile(source):
            try:
                with open(source, 'rb') as f:
                    saved = pickle.load(f)
                    if isinstance(saved, Pipeline):
                        self._name = name if name else saved.name()
                        self._transforms = saved.transforms()
                    else:
                        raise InvalidPipelineInputSource()
            except pickle.UnpicklingError:
                raise InvalidPipelineInputSource() from None
        else:
            raise InvalidPipelineInputSource()

    def name(self):
        return self._name

    def transforms(self):
        return self._transforms

    def __call__(self, image):
        for transform in self._transforms:
            image = transform(image)
        return image

    def save(self, filename=None):
        if not filename:
            filename = self._name + '.pipe'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
