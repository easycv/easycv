from functools import wraps

from easycv.pipeline import Pipeline


def auto_compute(decorated):
    @wraps(decorated)
    def wrapper(lazy_object, *args, **kwargs):
        lazy_object.compute(in_place=True)
        return decorated(lazy_object, *args, **kwargs)

    return wrapper


class Collection:
    def __init__(self, pending=None, lazy=False):
        self._lazy = lazy

        if pending is not None:
            if isinstance(pending, Pipeline):
                self._pending = pending.copy()
                self._pending._name = "pending"
            else:
                self._pending = Pipeline([pending], name="pending")
        else:
            self._pending = Pipeline([], name="pending")

    @property
    def pending(self):
        """
        Returns all pending transforms/pipelines.

        :return: Pipeline containing pending operations
        :rtype: :class:`~eascv.pipeline.Pipeline`
        """
        return self._pending

    def compute(self, inplace=False):
        pass
