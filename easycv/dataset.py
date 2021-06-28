from easycv import List
from easycv.collection import Collection, auto_compute
from easycv.errors.dataset import InvalidClass, NoClassesGiven

import os


class Dataset(Collection):
    def __init__(self, path, pipeline=None, recursive=False):
        super().__init__(pending=pipeline)
        self.path = path
        self.dataset = {}
        self._size = -1
        self.setup(path, recursive)

    def setup(self, path, recursive):
        classes = next(os.walk(path))[1]
        if not classes:
            raise NoClassesGiven()
        for clas in classes:
            self.dataset[clas] = List(os.path.join(path, clas), lazy=True, recursive=recursive)

    @property
    def classes(self):
        return list(self.dataset.keys())

    @auto_compute
    def get(self, key):
        if key not in self.dataset:
            raise InvalidClass(key)
        return self.dataset[key]

    def size(self):
        if self._size == -1:
            self._size = sum([len(self.dataset[clas]) for clas in self.dataset])
        return self._size

    @auto_compute
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

    def compute(self, in_place=True, parallel=False):
        if not in_place:
            dataset = {}
        for clas in list(self.dataset):
            if in_place:
                self.dataset[clas].compute(in_place=in_place, parallel=parallel)
                if not len(self.dataset[clas]):
                    self.dataset.pop(clas)
            else:
                dataset[clas] = self.dataset[clas].compute(in_place=in_place, parallel=parallel)
                if not len(self.dataset[clas]):
                    dataset[clas].pop(clas)
        if not in_place:
            return dataset

