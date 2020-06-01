Transforms
==========

Transforms are the most important part of EasyCV. This module contains all the transforms that EasyCV \
currently supports. It also enables some powerful functionalities that we'll discuss later on this page.

Introduction
-------------

All transforms inherit from the base class Transform witch takes care of most of the core functionality \
simplifying the process of creating new transforms.

Transforms can be used to modify an image (e.g. blur an image) or to extract valuable information \
(e.g. count the number of faces in the image). Any of the mentioned types of transform can be applied \
in the same way. The only difference between them is their outputs.


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