# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:30:05 2019

@author: Jongmin Sung

custom_funcs.py
In projectname/projectname/custom_funcs.py, we can put in custom code that 
gets used across more than notebook. One example would be downstream data 
preprocessing that is only necessary for a subset of notebooks.

# custom_funcs.py

"""

from .custom_funcs import add_one, capital_case


def test_add_one():
    assert add_one(1) == 2


def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'