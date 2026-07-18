
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
		The target nucleus.  The IAEA monitor, IRDFF and TENDL residual-product
		libraries also carry natural elements, e.g. 'natEl'. 

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
	ENDF/B-VIII.1
	>>> rx = ci.Reaction('226RA(n,x)225RA')
	>>> print(rx.library.name)
	TENDL-2025
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
				libs = ['irdff','endf','iaea','tendl','tendl_n_rp']
			elif self.incident in ['p','d','a']:
				libs = ['iaea','tendl_'+self.incident+'_rp']
			else:
				libs = ['iaea']
			for lb in libs:
				self.library = Library(lb)
				if lb==libs[-1]:
					self._check(True)
				elif self._check():
					break
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

		try:
			if 'nat' not in self.target:
				tg = Isotope(self.target).TeX
			else:
				tg = r'$^{nat}$'+self.target[3:].title()
			prd = Isotope(self.product).TeX if self.product else ''

			self.TeX = '{0}({1},{2}){3}'.format(tg, self.incident, self.outgoing, prd)

		except:
			self.TeX = reaction_name

	def _check(self, err=False):
		c = len(self.library.search(*self._rx))==1
		if err and not c:
			raise ValueError('Reaction '+self.name+' not found or not unique.')
		return c

	def __str__(self):
		return self.name
		
	def interpolate(self, energy):
		"""Interpolated cross section

		Linear interpolation of the reaction cross section along the
		input energy grid.  Energies outside the evaluated grid return 0
		rather than an extrapolation.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		Returns
		-------
		cross_section : np.ndarray
			Interpolated cross section, in mb.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)', 'IRDFF')
		>>> print(rx.interpolate(0.5))
		161.41646650941306
		>>> print(rx.interpolate([0.5, 1.0, 5.0]))
		[161.41646651 171.81486757 8.8822]

		""" 

		if self._interp is None:
			kind = 'linear'
			fv = 0.0
			i = 0
			if self.library.name.lower().startswith('tendl'):
				kind = 'quadratic'
				fv = 0.0
				ix = np.where(self.xs>0)[0]
				if len(ix)>0:
					i = max((ix[0]-1, 0))
					if len(self.xs)-i<5:
						kind = 'linear'
			self._interp = interp1d(self.eng[i:], self.xs[i:], bounds_error=False, fill_value=fv, kind=kind)
		_interp = self._interp(energy)
		return np.where(_interp>0, _interp, 0.0)

	def interpolate_unc(self, energy):
		"""Uncertainty in interpolated cross section

		Linear interpolation of the uncertainty in the reaction cross section
		along the input energy grid, for libraries where uncertainties are provided.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		Returns
		-------
		unc_cross_section : np.ndarray
			Uncertainty in the interpolated cross section, in mb.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)', 'IRDFF')
		>>> print(rx.interpolate_unc(0.5))
		3.9542683715745546
		>>> print(rx.interpolate_unc([0.5, 1.0, 5.0]))
		[3.95426837 5.88023936 0.4654]

		""" 

		if self._interp_unc is None:
			self._interp_unc = interp1d(self.eng, self.unc_xs, bounds_error=False, fill_value=0.0)
		return self._interp_unc(energy)
		
	def integrate(self, energy, flux, unc=False):
		"""Reaction flux integral

		Integrate the product of the cross section and flux along the input energy grid.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		flux : array_like
			Incident particle flux as a function of the input energy grid.

		unc : bool, optional
			If `True`, returns the both the flux integral and the uncertainty. If `False`,
			just the flux integral is returned. Default `False`.

		Returns
		-------
		xs_integral : float or tuple
			Reaction flux integral if `unc=False` (default), or reaction flux integral
			and uncertainty, if `unc=True`.

		Examples
		--------
		>>> rx = ci.Reaction('Ni-58(n,p)')
		>>> eng = np.linspace(1, 5, 20)
		>>> phi = np.ones(20)
		>>> print(rx.integrate(eng, phi))
		885.5690635272...
		>>> print(rx.integrate(eng, phi, unc=True))
		(885.5690635272..., ...)

		"""

		E = np.asarray(energy, dtype=float)
		phi = np.asarray(flux, dtype=float)
		dE = self._bin_widths(E)
		phisig = phi*self.interpolate(E)
		if unc:
			unc_phisig = phi*self.interpolate_unc(E)
			return np.sum(phisig*dE), np.sum(unc_phisig*dE)
		return np.sum(phisig*dE)

	@staticmethod
	def _bin_widths(E):
		# Treat E values as bin centers; bin edges sit at midpoints between centers,
		# endpoint bins get the one-sided width. For uniform grids this returns a
		# constant dE, so flux-averages reduce to a PMF-style sum(phi*sigma)/sum(phi)
		# and the result is independent of the (then-cancelling) dE. For histogram
		# fluxes from Stack.get_flux this avoids the trapezoidal endpoint
		# under-weighting that halves the contribution of bin 0 / bin -1 spikes.
		E = np.asarray(E, dtype=float)
		n = E.size
		if n < 2:
			return np.ones_like(E)
		dE = np.empty(n)
		dE[1:-1] = 0.5*(E[2:] - E[:-2])
		dE[0]    = E[1] - E[0]
		dE[-1]   = E[-1] - E[-2]
		return dE

	def average(self, energy, flux, unc=False):
		"""Flux averaged reaction cross section

		Calculates the flux-weighted average reaction cross section, using the
		input flux and energy grid.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		flux : array_like
			Incident particle flux as a function of the input energy grid.

		unc : bool, optional
			If `True`, returns the both the flux average cross section and the uncertainty. If `False`,
			just the average cross section is returned. Default `False`.

		Returns
		-------
		average_xs : float or tuple
			Flux-averaged reaction cross section if `unc=False` (default), or average
			and uncertainty, if `unc=True`.

		Examples
		--------
		>>> rx = ci.Reaction('Ni-58(n,p)')
		>>> eng = np.linspace(1, 5, 20)
		>>> phi = np.ones(20)
		>>> print(rx.average(eng, phi))
		210.3226525877...
		>>> print(rx.average(eng, phi, unc=True))
		(210.3226525877..., ...)

		"""

		E = np.asarray(energy, dtype=float)
		phi = np.asarray(flux, dtype=float)
		dE = self._bin_widths(E)
		phisig = phi*self.interpolate(E)
		denom = np.sum(phi*dE)
		if unc:
			unc_phisig = phi*self.interpolate_unc(E)
			return np.sum(phisig*dE)/denom, np.sum(unc_phisig*dE)/denom
		return np.sum(phisig*dE)/denom
		
	def plot(self, energy=None, label='reaction', title=False, **kwargs):
		"""Plot the cross section

		Plots the energy differential cross section.

		Parameters
		----------
		energy : array_like, optional
			Energy grid along which to plot the cross section.  If None, the
			energy grid provided by the library will be used. 

		label : str, optional
			Axes label.  If label='reaction', the label will be the reaction name.
			If 'library', it will be the name of the cross section library.
			If 'both', then the reaction name and library will be given.  If
			none of these options, pyplot will be called with `ax.plot(..., label=label)`.

		title : bool, optional
			Display the reaction name as the plot title.  Default, False.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)')
		>>> rx.plot(scale='loglog')
		>>> rx = ci.Reaction('35CL(n,p)')
		>>> f, ax = rx.plot(return_plot=True)
		>>> rx = ci.Reaction('35CL(n,el)')
		>>> rx.plot(f=f, ax=ax, scale='loglog')

		"""

		f, ax = _init_plot(**kwargs)

		if title:
			ax.set_title(self.TeX)

		if label is not None:
			if label.lower() in ['both','library','reaction']:
				label = {'both':'{0}\n({1})'.format(self.TeX, self.library.name),'library':self.library.name,'reaction':self.TeX}[label.lower()]

		unc_xs = None
		if energy is None:
			if self.library.name.lower().startswith('tendl'):
				eng = np.linspace(min(self.eng), max(self.eng), 801)
				xs = self.interpolate(eng)
			else:
				eng, xs = self.eng, self.xs
				if np.any(self.unc_xs>0):
					unc_xs = self.unc_xs
		else:
			eng, xs = np.asarray(energy), self.interpolate(energy)
			ux = self.interpolate_unc(energy)
			if np.any(ux>0):
				unc_xs = ux

		line, = ax.plot(eng, xs, label=label)
		if unc_xs is not None:
			ax.fill_between(eng, xs+unc_xs, xs-unc_xs, facecolor=line.get_color(), alpha=0.5)

		if self.library.name.lower().startswith('tendl'):
			wh = np.where((self.eng>=min(eng))&(self.eng<=max(eng)))
			elib = self.eng[wh]
			xslib = self.xs[wh]
			ax.plot(elib, xslib, ls='None', marker='o', color=line.get_color())

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel('Cross Section (mb)')

		if label:
			ax.legend(loc=0)

		return _draw_plot(f, ax, **kwargs)
		