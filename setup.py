#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)



setup(
    name='cellpose-napari',
    author='Carsen Stringer',
    author_email='stringerc@janelia.hhmi.org',
    license='BSD-3',
    url='https://github.com/carsen-stringer/cellpose-napari',
    description='a generalist algorithm for anatomical segmentation',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    use_scm_version=True,
    install_requires=requirements,
    setup_requires=['setuptools_scm', 'pytest-runner'],
    tests_require=['pytest', 'pytest-qt'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Framework :: napari',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
    ],
    entry_points={
        'napari.plugin': [
            'cellpose-napari = cellpose_napari',
        ],
    },
)
