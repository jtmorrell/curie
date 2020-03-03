from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

class Calibration(object):
	"""Calibration

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Attributes
	----------
	engcal : type
		Description of parameter

	effcal : type
		Description of parameter

	unc_effcal : type
		Description of parameter

	rescal : type
		Description of parameter


	Examples
	--------

	"""

	def __init__(self, filename=None, sources=None):
		pass

	def eng(self, channel, *cal):
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
		
	def eff(self, energy, *cal):
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
		
	def unc_eff(self, energy, cal=None, unc=None):
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
		
	def res(self, channel, *cal):
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
		
	def map_channel(self, energy, *cal):
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
		
	def calibrate(self, spectra):
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
		
	def saveas(self, filename):
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
		
	def plot_engcal(self, **kwargs):
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
		
	def plot_rescal(self, **kwargs):
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
		
	def plot_effcal(self, shelves=None, **kwargs):
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
		
	def plot(self, **kwargs):
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
		