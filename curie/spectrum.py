from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
import datetime as dtm

from .data import _get_connection

class Spectrum(object):
	"""Spectrum

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Attributes
	----------
	db : type
		Description

	cb : type
		Description

	filename : type
		Description

	directory : type
		Description

	shelf : type
		Description

	isotopes : type
		Description

	fit_config : type
		Description

	peaks : type
		Description


	Examples
	--------

	"""

	def __init__(self, filename, cb=None, shelf=None):
		pass

	def __str__(self):
		pass

	def __add__(self, other):
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
		
	def rebin(self, N_bins):
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
		
	def auto_calibrate(self):
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
		
	def attenuation_correction(self, cm, x=None, ad=None): # first compound is 'self'
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
		
	def geometry_correction(self):
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
		
	def multiplet(self, x, *args):
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
		
	def summarize(self):
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
		
	def plot(self, fit=True, labels=True, snip=False, xcalib=True, **kwargs):
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
		