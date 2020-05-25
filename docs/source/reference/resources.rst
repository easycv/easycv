Resources
======================

Resources are an easy way to manage large files that transforms need and are not included on easycv by default. \
Examples of such files are datasets and pre-trained models.

Creating a resource
--------------------
Resources are represented by a YAML file. You can manually create this file or use the built-in resource creator.

Using resource creator
^^^^^^^^^^^^^^^^^^^^^^
The following script creates a resource.

.. code-block:: python

    from easycv.resources import create_resource

    create_resource()

    >>> Resource name: test-resource
    >>> Number of files: 1
    >>> File 1 name: file1.txt
    >>> File 1 url: example.com/file1.txt
    >>> Downloading/hashing files...
    >>> Resource created successfully

Manually
^^^^^^^^
A resource can be represented by YAML file. We could create the resource described above with the following file:

.. code-block::

    files:
        - filename: file1.txt
          sha256: 0f7d45278...948e822da // file sha256 hash
          url: example.com/file1.txt

To add this resource to easycv, save the file as **test-resource.yaml** anc put it inside \
**easycv\\resources\\sources**.


Using resources
----------------
Resources can be used inside transforms to access big files that don't ship with EasyCV by default.

For example, if you want to create a transform that uses a pre-trained neural network you can create a resource that \
represents the weights file. Then on your transform, you require the resource, and EasyCV downloads the resource \
automatically when your transform is used.


Let's create a transform called TestTransform that uses the resource we created above. We can obtain a path to the \
file by using ``get_resource``.

.. code-block:: python

    from easycv.resources import get_resource
    from easycv.transforms.base import Transform

    class TestTransform(Transform):
        def process(self, image, **kwargs):
            file = get_resource("test-resource", "file1.txt")
            (...) # do something with the file and the image
            return image


Functions
----------------
.. automodule:: easycv.resources.creator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: easycv.resources.resources
   :exclude-members: load_resource_info
   :members:
   :undoc-members:
   :show-inheritance:
