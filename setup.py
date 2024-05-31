import os
import sys

from setuptools import setup, find_packages


setup(
    name = 'polycrystal',
    author = 'Donald E. Boyce',
    author_email = 'donald.e.boyce@gmail.com',
    description = 'polycrystal material modeling',
    classifiers = [
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
    packages = find_packages(),
    )
install_reqs = [
    'numpy',
    'scipy',
    'pytest',
]
