Installation
==============

You can install **EasyCV** with ``pip`` or from source.

.. note::
    **EasyCV** only supports Python 3 so make you sure you have it installed before proceeding.

Using Pip
---------
First, ensure that you have the latest pip version to avoid dependency errors::

   pip install --upgrade pip

Then install **EasyCV** and all its dependencies using `pip <https://pip.pypa.io/en/stable/>`_::

   pip install easycv

Install from Source
-------------------

To install EasyCV from source, clone the repository from `github
<https://github.com/easycv/easycv>`_::

    git clone https://github.com/Resi-Coders/easycv.git
    cd easycv
    pip install .

You can view the list of all dependencies within the ``install_requires`` field
of ``setup.py``.

Run Tests
-----------

Test EasyCV with ``pytest``. If you don't have ``pytest`` installed run::

    pip install pytest

Then to run all tests just run::

    cd easycv
    pytest .

