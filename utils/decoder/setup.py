#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension
import numpy


trie_module = Extension('_trie',
                        sources=['trie_wrap.cxx', 'trie.cpp'],
                        include_dirs=[numpy.get_include()],
                        )

setup (name = 'trie',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [trie_module],
       py_modules = ["trie"],
       )
