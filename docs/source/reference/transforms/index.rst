Transforms
==========

Transforms are the most important part of EasyCV. This module contains all the transforms that EasyCV \
currently supports. It also enables some powerful functionalities that we'll discuss later on this page.

Introduction
-------------

All transforms inherit from the base class Transform. The Transform class takes care of most of the core functionality \
simplifying the process of creating new transforms.

Transforms can be used to modify an image (e.g. blur an image) or to extract valuable information \
(e.g. count the number of faces in the image). Any of the mentioned types of transform can be applied \
in the same way. The only difference between them is their outputs.

Transform structure
-------------------

All transforms follow the same structure. They all inherit from Transform and must override the method ``process``. \
This method receives the image array and should return the result of applying the transform to the image. ``process`` \
also receives all the transform arguments, we will talk about arguments in detail later on this documentation.

Let's implement a simple transform that sets the first color channel to zero.

.. code-block:: python

    from easycv.transforms.base import Transform

    class FillChannel(Transform): # inherit from Transform

        def process(self, image, **kwargs): # override process
            image[:, :, 0] = 0
            return image


As you can see from the example above it's really simple to extend EasyCV. Now we can use our transform!

.. code-block:: python

    from easycv import Image

    img = Image("lenna.jpg")
    img.apply(FillChannel()).show() # Opens a popup window with the altered image


.. warning::
    The **process** method always receives the image as a numpy array not as an ``easycv.image.Image`` instance.


Arguments
----------

Some transforms depend on hyper-parameters or other configurations. EasyCV calls them arguments and they can \
specified be inside the transform class by assigning a dictionary containing the argument specifications to the \
variable ``arguments``. In this dictionary argument names are the keys and the values are the argument ``validators``. \
Validators enable easy and reliable argument validation/forwarding, more details about validators can be found \
:doc:`here <../validators>`.

EasyCV will check if the user is inserting the required arguments (arguments without a default value are \
considered required) and if the inserted values are valid. A helpful error message is generated automatically from the \
argument validator is generated in case of an error.

Let's add argument to the transform we created above. We'll add the argument ``fill_value`` to enable people to change \
the value we use to fill the channel! Since ``fill_value`` must be an 8-bit integer we'll use a ``Number`` validator \
with some restrictions applied.

Arguments are given to ``process`` through ``kwargs``. You can assume that process only runs if all arguments are \
valid and all required arguments have been filled.

.. note::
    The **process** method receives all the specified arguments on ``kwargs`` regardless of which arguments the user \
    enters (default values are used for this). There is an exception to this case we'll discuss later \
    (method-specific arguments).

.. code-block:: python

    from easycv.validators import Number

    class FillChannel(Transform):

        # define arguments
        arguments = {
            "fill_value": Number(min_value=0, max_value=255, only_integer=True, default=0),
        }

        def process(self, image, **kwargs):
            image[:, :, 0] = kwargs["fill_value"] # use the new argument instead of always zero
            return image


Now we can use the updated transform with the ``fill_value`` argument!

.. code-block:: python

    from easycv import Image

    img = Image("lenna.jpg")
    img.apply(FillChannel(fill_value=128)).show() # Let's fill the image channel with 128


But what happens if the argument is invalid? Let's check!

.. code-block:: python

    img.apply(FillChannel(fill_value="bad value")).show()

    >>> (Exception raised)
    >>> InvalidArgumentError: Invalid value for fill_value. Must be an integer between 0 and 255.


As we can see a helpful message is displayed warning the user that ``fill_value`` must be an integer between 0 \
and 255.

List of Transforms
------------------

.. toctree::
   :maxdepth: 2

   color
   detect
   draw
   edges
   filter
   noise
   selectors
   perspective
   spatial
   morphological

