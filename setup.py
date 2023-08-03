#!/usr/bin/env python
import sys
import os
from distutils.dist import Distribution

if 'setuptools' in sys.modules or any(
        s.startswith('bdist') for s in sys.argv) or any(
        s.startswith('develop') for s in sys.argv):
    from setuptools import setup as setup
    from setuptools import Extension
else:  # special case for runtests.py
    from distutils.core import setup as setup
    from distutils.extension import Extension

# Avoid installing setup_requires dependencies if the user just
# queries for information
if (any('--' + opt in sys.argv for opt in
        Distribution.display_option_names + ['help']) or
        'clean' in sys.argv):
    setup_requires = []
else:
    setup_requires = ['numpy']

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)
Operating System :: MacOS
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Programming Language :: Python
Topic :: Scientific/Engineering

"""
setup(
    name="visualizer",
    version="1.0.0",
    packages=['visualizer',
              'visualizer.data',
              'visualizer.visuals',
              ],
    #   scripts = [''],
    #
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        'h5py>=2.8.0',
        'jupyter>=1.0.0',
        'kaleido>=0.2.1',
        'matplotlib>=3.1.1',
        'numpy>=1.16.5',
        'pandas>=0.25.1',
        'plotly>=5.3.1',
        'scikit-learn>=0.21.3',
        'seaborn>=0.9.0',
        'statsmodels>=0.10.1',
        'tifffile>=2021.11.2',
        'pytest>=7.1.2'
    ],
    #
    # metadata for upload to PyPI
    author="Charles Zhou",
    author_email="zhouzc@uw.edu",
    description="Package that creates visualizations for imaging data",
    license="MIT",
    keywords="imaging microscopy neuroscience analysis",
    setup_requires=setup_requires,
    # setup_requires=['setuptools_cython'],
    platforms=["Linux", "Mac OS-X", "Windows"],
)
