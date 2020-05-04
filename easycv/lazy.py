from functools import wraps

from easycv.pipeline import Pipeline


def auto_compute(decorated):
    @wraps(decorated)
    def wrapper(lazy_object, *args, **kwargs):
        lazy_object.compute(in_place=True)
        return decorated(lazy_object, *args, **kwargs)

    return wrapper


class Lazy:
    def __init__(self, pending=None):
        if pending is not None:
            self._pending = pending.copy()
        else:
            self._pending = Pipeline([])

    def compute(self, inplace=False):
        pass
