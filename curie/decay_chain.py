from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import datetime as dtm

from .data import _get_connection

class DecayChain(object):
	"""DecayChain

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Attributes
	----------
	isotopes : type
		Description

	counts : type
		Description


	Examples
	--------

	"""

	def __init__(self, parent_isotope, units='s', R=None, A0=None, decay_time=None):
		pass

	def activity(self, isotope, t=None, units=None):
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
		
	def decays(self, isotope, t_start, t_stop, units=None):
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
		
	def read_db(self, db, spec_fnms=None, EoB=None):
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

	def read_csv(self, csv, spec_fnms=None, EoB=None):
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
		
	def fit_spectra(self, spectra, db=None, max_unc=0.15, EoB=None):
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
		
	def fit_R(self, isotope=None, unc=False, _update=True):
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
		
	def fit_A0(self, isotope=None, unc=False, _update=True):
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
		
	def plot(self, time=None, max_plot=None, max_label=10, **kwargs):
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
		