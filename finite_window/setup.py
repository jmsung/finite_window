# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:30:05 2019

@author: Jongmin Sung

setup.py

"""

from setuptools import setup
from inspect import currentframe, getframeinfo
from pathlib import Path

fname = getframeinfo(currentframe()).filename
path = Path(fname).resolve().parent.parent

# Get the long description from the README file
with open(path.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='finite_winow',
    version='0.1.0',
    author='Jongmin Sung',
    author_email='jongmin.sung@gmail.com',
    license=path.joinpath('LICENSE.MD'),
    description='Finite_window.',
    long_description=long_description,
)
