from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from .isotope import Isotope
from .library import Library

class Reaction(object):
	"""Cross section data for nuclear reactions

	Contains reaction cross sections as a function of incident energy,
	and some useful methods for manipulating cross section data, such as
	flux-averages, integrated cross-sections, and interpolation.  All 
	cross sections (and uncertainties) are in mb, and all energies are
	in MeV.
	
	Parameters
	----------
	reaction_name : str
		Name of the reaction, in nuclear reaction notation. E.g. '115IN(n,g)',
		'235U(n,f)', '139LA(p,x)134CE', 'Ra-226(n,2n)Ra-225', 'Al-27(n,a)', etc.

	library : str, optional
		Name of the library to use, or 'best' (default).

	Attributes
	----------
	target : str
		The target nucleus.  Some libraries support natural elements, e.g. 'natEl'. 

	incident : str
		Incident particle. E.g. 'n', 'p', 'd'. 

	outgoing : str
		Outgoing particle, or reaction shorthand.  E.g. '2n', 'd', 'f', 'inl', 'x'.
		Will always be 'x' for (TENDL) residual product libraries. 

	product : str
		The product isotope. 

	eng : np.ndarray
		Incident particle energy, in MeV. 

	xs : np.ndarray
		Reaction cross section, in mb. 

	unc_xs : np.ndarray
		Uncertainty in the cross section, in mb.  If not provided by the
		library, default is zeros of same shape as xs. 

	name : str
		Name of the reaction in nuclear reaction notation. 

	library : ci.Library
		Nuclear reaction library.  printing `rx.library.name` will give the
		name of the library. 

	TeX : str
		LaTeX formatted reaction name. 


	Examples
	--------
	>>> rx = ci.Reaction('226RA(n,2n)')
	>>> print(rx.library.name)
	ENDF/B-VII.1
	>>> rx = ci.Reaction('226RA(n,x)225RA')
	>>> print(rx.library.name)
	TENDL-2015
	>>> rx = ci.Reaction('115IN(n,inl)')
	>>> print(rx.library.name)
	IRDFF-II

	"""

	def __init__(self, reaction_name, library='best'):
		self.target, p = tuple(reaction_name.split('('))
		p, self.product = tuple(p.split(')'))
		self.incident, self.outgoing = tuple(p.split(','))
		self.incident, self.outgoing = self.incident.lower(), self.outgoing.lower()
		self._rx = [self.target, self.incident, self.outgoing, self.product]
		self.name = reaction_name

		if library.lower()=='best':
			if self.incident=='n':
				for lb in ['irdff','endf','iaea','tendl','tendl_n_rp']:
					self.library = Library(lb)
					if lb=='tendl_n_rp':
						self._check(True)
					elif self._check():
						break
			elif self.incident in ['p','d']:
				for lb in ['iaea','tendl_'+self.incident+'_rp']:
					self.library = Library(lb)
					if lb=='tendl_d_rp':
						self._check(True)
					elif self._check():
						break
			else:
				self.library = Library('iaea')
				self._check(True)
		else:
			self.library = Library(library)
			self._check(True)

		self.name = self.library.search(*self._rx)[0]
		q = self.library.retrieve(*self._rx)
		self.eng = q[:,0]
		self.xs = q[:,1]
		if q.shape[1]==3:
			self.unc_xs = q[:,2]
		else:
			self.unc_xs = np.zeros(len(self.xs))
		self._interp = None
		self._interp_unc = None

		if 'nat' not in self.target:
			tg = Isotope(self.target).TeX
		else:
			tg = r'$^{nat}$'+self.target[3:].title()
		prd = Isotope(self.product).TeX if self.product else ''

		self.TeX = '{0}({1},{2}){3}'.format(tg, self.incident, self.outgoing, prd)

	def _check(self, err=False):
		c = len(self.library.search(*self._rx))==1
		if err and not c:
			raise ValueError('Reaction '+self.name+' not found or not unique.')
		return c

	def __str__(self):
		return self.name
		
	def interpolate(self, energy):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		Returns
		-------
		cross_section : np.ndarray
			Description

		Examples
		--------

		""" 

		if self._interp is None:
			self._interp = interp1d(self.eng, self.xs, bounds_error=False, fill_value=0.0)
		return self._interp(energy)

	def interpolate_unc(self, energy):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		Returns
		-------
		unc_cross_section : np.ndarray
			Description

		Examples
		--------

		""" 

		if self._interp_unc is None:
			self._interp_unc = interp1d(self.eng, self.unc_xs, bounds_error=False, fill_value=0.0)
		return self._interp_unc(energy)
		
	def integrate(self, energy, flux, unc=False):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		flux : array_like
			Description of x

		unc : bool, optional
			Description of x

		Returns
		-------
		xs_integral : np.ndarray
			Description

		Examples
		--------

		"""

		E = np.asarray(energy)
		phisig = np.asarray(flux)*self.interpolate(E)
		if unc:
			unc_phisig = np.asarray(flux)*self.interpolate_unc(E)
			return np.sum(0.5*(E[1:]-E[:-1])*(phisig[:-1]+phisig[1:])), np.sum(0.5*(E[1:]-E[:-1])*(unc_phisig[:-1]+unc_phisig[1:]))
		return np.sum(0.5*(E[1:]-E[:-1])*(phisig[:-1]+phisig[1:]))
		
	def average(self, energy, flux, unc=False):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		flux : array_like
			Description of x

		unc : bool, optional
			Description of x

		Returns
		-------
		average_xs : np.ndarray
			Description

		Examples
		--------

		"""

		E, phi = np.asarray(energy), np.asarray(flux)
		phisig = phi*self.interpolate(E)
		dE = E[1:]-E[:-1]
		if unc:
			unc_phisig = np.asarray(flux)*self.interpolate_unc(E)
			return np.sum(0.5*dE*(phisig[:-1]+phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:])), np.sum(0.5*dE*(unc_phisig[:-1]+unc_phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:]))
		return np.sum(0.5*dE*(phisig[:-1]+phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:]))
		
	def plot(self, energy=None, label='reaction', title=False, **kwargs):
		""" Description

		...

		Parameters
		----------
		energy : array_like, optional
			Description 

		label : str, optional
			Description of x

		title : bool, optional
			Description

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------

		"""

		f, ax = _init_plot(**kwargs)

		if title:
			ax.set_title(self.TeX)

		if label is not None:
			if label.lower() in ['both','library','reaction']:
				label = {'both':'{0}\n({1})'.format(self.TeX, self.library.name),'library':self.library.name,'reaction':self.TeX}[label.lower()]

		unc_xs = None
		if energy is None:
			eng, xs = self.eng, self.xs
			if np.any(self.unc_xs>0):
				unc_xs = self.unc_xs
		else:
			eng, xs = np.asarray(energy), self.interpolate(energy)
			ux = self.intepolate_unc(energy)
			if np.any(ux>0):
				unc_xs = ux

		line, = ax.plot(eng, xs, label=label)
		if unc_xs is not None:
			ax.fill_between(eng, xs+unc_xs, xs-unc_xs, facecolor=line.get_color(), alpha=0.5)

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel('Cross Section (mb)')

		if label:
			ax.legend(loc=0)

		return _draw_plot(f, ax, **kwargs)
		