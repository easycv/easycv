# author: Resi Coders
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# Get the long description from the README file
with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="easycv",
    version="0.1.0",
    description="Cv is a computer vision library built for learning purposes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Resi-Coders/easycv",
    author="Resi Coders",
    author_email="to-do@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="easycv computer vision ai",
    packages=find_packages(),
    setup_requires=["setuptools>=38.6.0"],
    install_requires=[
        "numpy",
        "pillow",
        "requests",
        "scipy",
        "matplotlib",
        "opencv-python",
        "scikit-image",
    ],
)
