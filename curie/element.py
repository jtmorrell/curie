from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from .data import _get_connection

class Element(object):
	"""Element

	...
	
	Parameters
	----------
	element : str
		Description of parameter `x`.

	Attributes
	----------

	name : type
		Description of attribute

	Z : type
		Description of attribute

	mass : type
		Description of attribute

	isotopes : type
		Description of attribute

	abundances : type
		Description of attribute

	density : type
		Description of attribute

	mass_coeff : type
		Description of attribute

	mass_coeff_en : type
		Description of attribute


	Examples
	--------

	"""

	def __init__(self, element):
		pass

	def __str__(self):
		pass

	def attenuation(self, energy, x=1.0, ad=None):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		
	def dEdx(self, energy, particle='p'):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		
	def range(self, energy, particle='p'):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		
	def plot_mass_coeff(self, energy=None, **kwargs):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		
	def plot_mass_coeff_en(self, energy=None, **kwargs):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		
	def plot_S(self, compound, energy=None, **kwargs):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		pass
		