from copy import copy

import easycv.image
from easycv.errors import MissingArgumentError


class Operation:
    arguments = None  # Validators for arguments
    outputs = None  # Validators for outputs

    _args = None  # Argument values

    @property
    def args(self):
        """
        Returns a dictionary containing all names/values of the operation arguments. If the \
        operation isn't initialized yet this only returns the arguments provided when creating \
        the transform. Default arguments are only added on `initialize`.

        :return: Dictionary with all arguments and values.
        :rtype: :class:`dictionary`
        """
        return self._args

    def initialize(self, index=None, forwarded=()):
        """
        Initializes the operation. Initializing is verifying if the operation has all the \
        required arguments to run and adding the default values for the arguments that were not \
        provided or forwarded. If some required arguments are missing a \
        :class:`~easycv.errors.MissingArgumentError` is raised.

        :param index: Index of the operation, defaults to None
        :type index: :class:`int`, optional
        :param forwarded: List of arguments forwarded to the operation, defaults to no forwards
        :type forwarded: :class:`list`/:class:`tuple`, optional
        """

        for arg in self.arguments:
            if arg not in self._args:
                if self.arguments[arg].default is None and arg not in forwarded:
                    raise MissingArgumentError(arg, index=index)

                validator = self.arguments[arg]
                self._args[arg] = validator.default

    def can_be_forwarded(self, arg_name, validator):
        """
        Checks if the argument can be forwarded into this operation. An argument can be forwarded \
        if and only if it is one of the operation arguments, wasn't specified manually and the \
        validators are compatible.

        :param arg_name: Name of the argument to check
        :type arg_name: :class:`str`
        :param validator: Validator of the argument to check
        :type validator: :class:`~easycv.validators.validator`
        :return: *True* if the argument can be forwarded into the operation *False* otherwise
        :rtype: :class:`bool`
        """

        if arg_name in self.arguments and arg_name not in self._args:
            return self.arguments[arg_name].accepts(validator)
        return False

    def run(self, image, forwarded=None):
        """
        This method applies the operation to the given image array and returns the result.
        All operations must to override this method.

        :param image: Image represented as an array
        :type image: :class:`~numpy:numpy.ndarray`
        :param forwarded: Forwarded arguments, defaults to None
        :type forwarded: :class:`dict`, optional
        :return: The image as an array after the operation
        :rtype: :class:`~numpy:numpy.ndarray`
        """

    def apply(self, image, in_place=False):
        """
        This method returns the image after applying this operation. If image is an array it \
        returns the altered array. If it is an `Image` object returns a new `Image` with this \
        operation applied. If in_place id *True* it alters the given object instead of creating a \
        new one.

        :param image: Image object or image as an array
        :type image: :class:`~easycv.image.Image`/:class:`~numpy:numpy.ndarray`
        :param in_place: `True` to change the *image* object, `False` to return a new one with \
        the transform applied, defaults to `False`
        :type in_place: :class:`bool`, optional
        :return: The image after the operation
        :rtype: :class:`~easycv.image.Image`/:class:`~numpy:numpy.ndarray`
        """
        if isinstance(image, easycv.image.Image):
            return image.apply(self, in_place=in_place)
        else:
            return self.run(image)

    def copy(self):
        return copy(self)
