# author: EasyCV Developers
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name="easycv",
    packages=["easycv"],
    version="0.1.1",
    license="MIT",
    description="Computer Vision made easy",
    author="EasyCV Developers",
    url="https://github.com/easycv/easycv",
    download_url="https://github.com/easycv/easycv/archive/v0.1.1.tar.gz",
    keywords=["easycv", "computer vision", "python", "ai"],
    install_requires=[
        "numpy",
        "pillow",
        "requests",
        "scipy",
        "matplotlib",
        "opencv-python",
        "scikit-image",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
