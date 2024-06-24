from easycv import List
from easycv.collection import Collection, auto_compute
from easycv.errors.dataset import InvalidClassError, NoClassesGivenError

import os


class Dataset(Collection):
    def __init__(self, source, pipeline=None, recursive=False, lazy=False):
        super().__init__(pending=pipeline)
        self._size = -1
        if isinstance(source, dict):
            self.dataset = source
            self.size()
        else:
            self.dataset = {}
            self.load_classes(source, recursive, lazy)

    def load_classes(self, path, recursive, lazy):
        classes = next(os.walk(path))[1]
        if not classes:
            raise NoClassesGivenError()
        for clas in classes:
            self.dataset[clas] = List(os.path.join(path, clas), lazy=lazy, recursive=recursive)
        if not lazy:
            self._size = self.size()

    @property
    def classes(self):
        return list(self.dataset.keys())

    @auto_compute
    def get(self, key):
        if key not in self.dataset:
            raise InvalidClassError(key)
        return self.dataset[key]

    def __getitem__(self, key):
        return self.get(key)

    def size(self):
        if self._size == -1:
            self._size = sum([len(self.dataset[clas]) for clas in self.dataset])
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
            return Dataset(dataset)

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
            return Dataset(dataset)


