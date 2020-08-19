"""
curie
=====

Provides
  1. Classes for plotting and fitting gamma ray spectra
  2. Access to data from evaluated nuclear reaction libraries
  3. Photon attenuation data
  4. Charged particle stopping power calculations
  5. Generalized Bateman equation solver
  6. Atomic and nuclear structure/decay data

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a users guide located on
`the Curie homepage <https://jtmorrell.github.io/curie/build/html/index.html>`_.

The docstring examples assume that `curie` has been imported as `ci`::

    >>> import curie as ci


Code snippets are indicated by three greater-than signs::

  >>> sp = ci.Spectrum('spectrum.Spe')
  >>> sp.plot()

Use the built-in ``help`` function to view a function or class's docstring::

  >>> help(ci.Spectrum)




"""

from .data import _data_path, download
from .plotting import colormap, set_style
set_style('default')

from .isotope import Isotope
from .compound import Compound, COMPOUND_LIST
from .element import Element

from .spectrum import Spectrum
from .calibration import Calibration

from .decay_chain import DecayChain

from .library import Library
from .reaction import Reaction

from .stack import Stack

__version__ = '0.0.8'
__all__ = ['download', 'colormap', 'set_style', 
          'Isotope', 'Element', 'Compound', 
          'Spectrum', 'Calibration', 'DecayChain', 
          'Library', 'Reaction', 'Stack']