import os
import pickle


class Pipeline(object):
    def __init__(self, source, name=None):
        if type(source) == list:
            self._name = name if name else 'pipeline'
            self._transforms = source
        elif type(source) == str and os.path.isfile(source):
            with open(source, 'rb') as f:
                saved = pickle.load(f)
                if type(saved) == Pipeline:
                    self._name = name if name else saved.name()
                    self._transforms = saved.transforms()
        else:
            pass

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
