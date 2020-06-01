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

    class RemoveFirst(Transform): # inherit from Transform

        def process(self, image, **kwargs): # override process
            image[:, :, 0] = 0
            return image


As you can see from the example above it's really simple to extend EasyCV. Now we can use our transform!

.. code-block:: python

    from easycv import Image

    img = Image("lenna.jpg")
    img.apply(RemoveFirst()).show() # Opens a popup window with the altered image


.. warning::
    The **process** method always receives the image as a numpy array not as an ``easycv.image.Image`` instance.


Arguments
----------

List of Transforms
------------------

.. toctree::
   :maxdepth: 2

   color
   edges
   filter
   noise
   selectors
   spatial
   perspective
   detect