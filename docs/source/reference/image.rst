Image
======================

The Image module provides a class with the same name which is used to represent an image.

Images provide a way of loading images easily from multiple sources and a simple but powerful way \
of applying Transforms/Pipelines.

.. note::
    **Easycv** uses BGR color scheme in the internal representation. This makes the interface with \
    opencv easier and faster.

Examples
----------
Load Images
^^^^^^^^^^^^^^^^^
The following script loads and displays an image from a file and then from a url.

.. code-block:: python

    from easycv import Image

    img = Image("lenna.jpg").show()
    img_from_url = Image("www.example.com/lenna.jpg").show()

Apply transforms
^^^^^^^^^^^^^^^^^
The following script uses the image from the last example. It turns the image into grayscale
and then blurs it.

.. code-block:: python

    from easycv.transforms import Blur, Grayscale

    img = img.apply(Grayscale())
    img = img.apply(Blur(sigma=50))
    img.show()

Auto Compute
-------------
Lazy images are only loaded/computed when their update array data is needed. Methods that need \
the updated image need to have this decorator to ensure that the image is computed before their \
execution.

.. automodule:: cv.image
   :members: auto_compute

Image Class
------------
.. autoclass:: cv.image.Image
   :members:
   :undoc-members:
   :show-inheritance:
