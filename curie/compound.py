from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from .data import _get_connection

class Compound(object):
	"""Compound

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Attributes
	----------
	density : type
		Description of parameter

	mass_coeff : type
		Description of parameter

	mass_coeff_en : type
		Description of parameter


	Examples
	--------

	"""

	def __init__(self, compound):
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
		