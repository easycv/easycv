Pipeline
============

The :mod:`Pipeline` module provides a class with the same name which is used to represent a Pipeline.

**Pipelines** can be applied to Images exactly like :doc:`Transforms <transforms/index>`. They \
consist of a series of :doc:`Transforms <transforms/index>` applied in order. They can also be \
saved/loaded for later use.

Examples
----------
Create simple Pipeline
^^^^^^^^^^^^^^^^^^^^^^^
The following script creates a simple **pipeline** and applies it to a previously loaded \
**image**. Info on how to load images can be found :doc:`here <image>`.

.. code-block:: python

    from easycv import Pipeline
    from easycv.transforms import Blur, Grayscale

    pipeline = Pipeline([Grayscale(), Blur(sigma=50)])
    img = img.apply(pipeline)
    img.show()

This is the same as doing the following (applying each **transform** separately):

.. code-block:: python

    img = img.apply(Grayscale())
    img = img.apply(Blur(sigma=50))
    img.show()

Save and load Pipeline
^^^^^^^^^^^^^^^^^^^^^^^
The following script saves and loads the **pipeline** created in the last example.

.. code-block:: python

    pipeline.save(filename='example.pipe')

    loaded = Pipeline('example.pipe')
    img = img.apply(loaded)
    img.show()

.. note::
    If you are running **Easycv** inside a jupyter notebook there is no need to call `show()` \
    , the image will be displayed if you evaluate it.

Pipeline Class
---------------
.. automodule:: easycv.pipeline
   :members:
   :undoc-members:
   :show-inheritance: