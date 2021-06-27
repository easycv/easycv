from easycv import Image, List
from easycv.collection import Collection
from easycv.io import open_dataset
from easycv.errors.dataset import InvalidClass


class Dataset(Collection):
    def __init__(self, path, pipeline=None, recursive=False):
        super().__init__(pending=pipeline)
        self.path = path
        self.dataset = {}
        self._size = 0
        self.setup(path, recursive)

    def setup(self, path, recursive):
        dataset = open_dataset(path, recursive)
        for clas in dataset:
            self.dataset[clas] = List([Image(img) for img in dataset[clas]])
            self._size += len(self.dataset[clas])

    @property
    def classes(self):
        return list(self.dataset.keys())

    def get(self, key):
        if key not in self.dataset:
            raise InvalidClass(key)
        return self.dataset[key]

    def size(self):
        return self._size

    def details(self):
        for clas in self.dataset:
            print(clas+":" + str(len(self.dataset[clas])))

    def apply(self, operation, in_place=False, parallel=False):
        if not in_place:
            dataset = {}
        for clas in self.dataset:
            if in_place:
                self.dataset[clas].apply(operation, in_place=in_place, parallel=parallel)
            else:
                dataset[clas] = self.dataset[clas].apply(operation, in_place=in_place, parallel=parallel)

        if not in_place:
            return dataset