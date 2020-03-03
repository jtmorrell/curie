from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from .data import _get_connection

class Reaction(object):
	"""Reaction

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Attributes
	----------
	target : type
		Description 

	incident : type
		Description 

	outgoing : type
		Description 

	product : type
		Description 

	eng : type
		Description 

	xs : type
		Description 

	unc_xs : type
		Description 

	name : type
		Description 

	library : type
		Description 

	TeX : type
		Description 


	Examples
	--------

	"""

	def __init__(self, reaction_name, library='best'):
		pass

	def __str__(self):
		pass
		
	def interpolate(self, unc=False):
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
		
	def integrate(self, energy, flux, unc=False):
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
		
	def average(self, energy, flux, unc=False):
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
		
	def plot(self, label=None, title=True, E_lim=None, **kwargs):
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
		