import os
import pickle

from cv.transforms.base import Transform
from cv.errors.io import InvalidPipelineInputSource


class Pipeline(object):
    def __init__(self, source, name=None):
        if isinstance(source, list) and all([isinstance(x, (Transform, Pipeline)) for x in source]):
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

    def description(self, level=0, start=1):
        index = str(start) + ': ' if start > 1 else ''
        indent = '    ' + '|    ' * (level-1) if level > 1 else '    ' * level
        r = [indent + index + f'Pipeline ({self._name}) with {self.num_transforms()} transforms']
        for i, t in enumerate(self._transforms):
            if isinstance(t, Pipeline):
                r.append(t.description(level=level+1, start=i+1))
            else:
                indent = '    ' + '|    ' * level
                r.append(f'{indent}{i + 1}: {str(t)}')
        return '\n'.join(r)

    def num_transforms(self):
        num = 0
        for t in self._transforms:
            if isinstance(t, Pipeline):
                num += t.num_transforms()
            else:
                num += 1
        return num

    def transforms(self):
        return self._transforms

    def __eq__(self, other):
        return isinstance(other, Pipeline) and self.name() == other.name() \
               and self.num_transforms() == other.num_transforms() \
               and all(t1 == t2 for t1, t2 in zip(self.transforms(), other.transforms()))

    def __str__(self):
        return self.description()

    def __repr__(self):
        return str(self)

    def __call__(self, image):
        for transform in self._transforms:
            image = transform(image)
        return image

    def save(self, filename=None):
        if not filename:
            filename = self._name + '.pipe'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
