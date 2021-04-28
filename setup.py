#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


install_deps = ['napari-plugin-engine>=0.1.4',
                'cellpose',
                'imagecodecs']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cellpose-napari',
    author='Carsen Stringer',
    author_email='stringerc@janelia.hhmi.org',
    license='BSD-3',
    url='https://github.com/carsen-stringer/cellpose-napari',
    description='a generalist algorithm for anatomical segmentation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    use_scm_version=True,
    install_requires=install_deps,
    setup_requires=['setuptools_scm', 'pytest-runner'],
    tests_require=['pytest', 'pytest-qt'],
    extras_require={
      "docs": [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
        'sphinx-prompt',
        'sphinx-autodoc-typehints',
      ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Framework :: napari',
    ],
    entry_points={
        'napari.plugin': [
            'cellpose-napari = cellpose_napari',
        ],
    },
)
