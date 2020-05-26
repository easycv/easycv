from easycv.collection import Collection
import inspect


class Output(Collection):
    def __init__(self, image, pending):
        self._image = image
        self._outputs = pending.outputs
        self._computed = False
        super().__init__(pending=pending)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Output(fields=[{}], computed={})".format(
            ", ".join(self._outputs), self._computed
        )

    def __getattr__(self, item):
        if item in self._outputs and inspect.stack()[1][3] != "getattr_paths":
            self.compute()
            return self._outputs[item]

    def __dir__(self):
        return list(super().__dir__()) + [str(k) for k in self.fields] + ["a"]

    def compute(self, inplace=True):
        output = self._outputs
        if not self._computed:
            output = self._pending(self._image)

        if inplace:
            self._outputs = output
            self._computed = True
        else:
            return output

    @property
    def fields(self):
        return list(self._outputs)
