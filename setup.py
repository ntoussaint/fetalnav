#!/usr/bin/env python
"""fetalnav : Fetal Region Detection using PyTorch and Soft Proposal Networks.

"""

import os
from setuptools import setup, find_packages
from codecs import open

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'license.txt')) as f:
    package_license = f.read()

setup(
    name="fetalnav",
    version="1.0.0",
    description="Fetal Region Detection using PyTorch",
    url="https://github.com/ntoussaint/fetalnav",
    author="Nicolas Toussaint",
    author_email="nicolas.toussaint@gmail.com",
    # Requirements
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=requirements,
    setup_requires=requirements,
    dependency_links=[],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=["build", "examples", "doc"]),
    package_data={'fetalnav': ['data/*.tar', 'data/*.mhd', 'data/*.raw']},
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # What does your project relate to?
    keywords='PyTorch deeplearning biomedical torchvision',
    long_description=long_description,
    license=package_license,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
