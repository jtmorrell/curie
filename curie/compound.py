from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .data import _get_connection
from .plotting import _init_plot, _draw_plot
from .element import ELEMENTS, Element

compound_list = list(map(str, pd.read_sql('SELECT compound FROM compounds', _get_connection('ziegler'))['compound']))

class Compound(object):
	"""Compound

	...
	
	Parameters
	----------
	compound : str
		Description of parameter `x`.

	weights : dict, str or pandas.DataFrame, optional
		Description

	density : float
		Description

	Attributes
	----------
	name : str
		Description

	weights : pandas.DataFrame
		Description

	density : float
		Density of the compound in g/cm^3. The density is used in
		calculations of charged particle dEdx and photon attenuation, so
		you can assign a new density using `cm.density = new_density` if
		needed, or using the `density` keyword in either of those functions.

	elements : list of ci.Element
		Elements

	mass_coeff : pandas.DataFrame
		Table of mass-attenuation coefficients as a function of photon
		energy, from the NIST XCOM database.  Energies are in keV, and
		mass-attenuation coefficients, or mu/rho, are given in cm^2/g.
		DataFrame columns are 'energy', 'mu' and 'mu_en' for the 
		mass-energy absorption coefficient.



	Examples
	--------

	"""

	def __init__(self, compound, weights=None, density=None):
		self.name = compound
		self.density = None

		if compound in compound_list:
			df = pd.read_sql('SELECT * FROM compounds WHERE compound="{}"'.format(compound), _get_connection('ziegler'))
			self.density = df['density'][0]

			wts = df['weights'][0].split(',')
			elements = [str(i.split(':')[0]) for i in wts]
			atom_weights = np.array([float(i.split(':')[1]) for i in wts])
			self._set_weights(elements, atom_weights=atom_weights)
			
		else:
			elements = []
			for el_gp in [i for i in re.split('[0-9]+|\\.', compound) if i]:
				for s in [i for i in re.split('([A-Z])', el_gp) if i]:
					if s.upper()==s:
						elements.append(s)
					else:
						elements[-1] += s

			if all([e in ELEMENTS for e in elements]):
				wts = re.split('|'.join(sorted(elements, key=lambda i:-len(i))), compound)
				atom_weights = np.array([float(wts[n+1]) if wts[n+1] else 1.0 for n,e in enumerate(elements)])
				self._set_weights(elements, atom_weights=atom_weights)

		if weights is not None:

			if type(weights)==dict:
				elements = [e for e in weights]
				wts = np.array([weights[e] for e in elements], dtype=np.float64)
				if wts[0]>0:
					self._set_weights(elements, atom_weights=wts)
				else:
					self._set_weights(elements, mass_weights=np.abs(wts))

			elif type(weights)==str:
				if weights.endswith('.json'):
					elements, wts = self._read_json(weights)

				elif weights.endswith('.csv'):
					elements, wts = self._read_csv(weights)

				else:
					elements = [str(i.split(':')[0]) for i in weights.split(',')]
					wts = np.array([float(i.split(':')[1]) for i in weights.split(',')])

				if wts[0]>0:
					self._set_weights(elements, atom_weights=wts)
				else:
					self._set_weights(elements, mass_weights=np.abs(wts))

			elif type(weights)==pd.DataFrame:
				if all([i in weights.columns for i in ['element', 'Z', 'atom_weight', 'mass_weight']]):
					self.weights = weights

				else:
					elements, wts = list(weights['element']), weights['weight'].to_numpy()

					if wts[0]>0:
						self._set_weights(elements, atom_weights=wts)
					else:
						self._set_weights(elements, mass_weights=np.abs(wts))


		if density is not None:
			self.density = density

		self.elements = [Element(el) for el in self.weights['element']]

		if self.density is None and len(self.weights)==1:
			self.density = self.elements[0].density

		E = np.unique(np.concatenate([el.mass_coeff['energy'].to_numpy() for el in self.elements]))
		mu = np.average([el.mu(E) for el in self.elements], weights=self.weights['mass_weight'], axis=0)
		mu_en = np.average([el.mu_en(E) for el in self.elements], weights=self.weights['mass_weight'], axis=0)
		self.mass_coeff = pd.DataFrame({'energy':E,'mu':mu,'mu_en':mu_en})
		self._mc_interp, self._mc_en_interp = None, None


	def _set_weights(self, elements, atom_weights=None, mass_weights=None):
		amu = pd.read_sql('SELECT * FROM weights', _get_connection('ziegler'))
		Zs = [ELEMENTS.index(el) for el in elements]

		if mass_weights is None:
			mass_weights = np.array([amu[amu['Z']==z]['amu'][z-1]*atom_weights[n] for n,z in enumerate(Zs)])
		elif atom_weights is None:
			atom_weights = np.array([mass_weights[n]/amu[amu['Z']==z]['amu'][z-1] for n,z in enumerate(Zs)])

		atom_weights, mass_weights = atom_weights/np.sum(atom_weights), mass_weights/np.sum(mass_weights)
		self.weights = pd.DataFrame({'element':elements, 'Z':Zs, 'atom_weight':atom_weights, 'mass_weight':mass_weights}, 
								columns=['element','Z','atom_weight','mass_weight'])

	def _read_json(self, filename):
		import json

		js = json.loads(open(filename).read())[self.name]
		elements = [el for el in js]
		return elements, np.array([js[el] for el in elements])

	def _read_csv(self, filename):
		df = pd.read_csv(filename, header=0, names=['compound', 'element', 'weight']).fillna(method='ffill')
		df = df[df['compound']==self.name]

		return list(df['element']), np.array(df['weight'], dtype=np.float64)

	def __str__(self):
		return self.name

	def mu(self, energy):
		if self._mc_interp is None:
			self._mc_interp = interp1d(np.log(self.mass_coeff['energy']), np.log(self.mass_coeff['mu']), bounds_error=False, fill_value='extrapolate')
		return np.exp(self._mc_interp(np.log(energy)))

	def mu_en(self, energy):
		if self._mc_en_interp is None:
			self._mc_en_interp = interp1d(np.log(self.mass_coeff['energy']), np.log(self.mass_coeff['mu_en']), bounds_error=False, fill_value='extrapolate')
		return np.exp(self._mc_en_interp(np.log(energy)))

	def attenuation(self, energy, x, density=None):
		"""Photon attenuation in matter

		Calculate the attenuation factor I(x)/I_0 = e^(-mu*x) for a given
		photon energy (in keV) and slab thickness (in cm).

		Parameters
		----------
		energy : array_like
			Incident photon energy in keV.

		x : float
			Thickness of slab of given element, in cm. 

		density : float, optional
			Density of the element in g/cm^3.  Default behavior is to
			use `Element.density`.

		Returns
		-------
		numpy.ndarray
			The slab attenuation factor as an absolute number (i.e. from 0 to 1).
			E.g. if the incident intensity is I_0, the transmitted intensity I(x) 
			is I_0 times the attenuation factor.

		Examples
		--------
		>>> el = ci.Element('Fe')
		>>> print(el.attenuation(511, x=0.3))
		0.821621630674751
		>>> print(el.attenuation(300, x=0.5, density=8))
		0.6442940871813587

		"""

		energy = np.asarray(energy, dtype=np.float64)
		x = np.asarray(x, dtype=np.float64)
		if density is None:
			density = self.density

		return np.exp(-self.mu(energy)*x*density)
		
	def S(self, energy, particle='p', density=None):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		energy = np.asarray(energy, dtype=np.float64)
		if density is None:
			density = self.density

		return np.average([el.S(energy, particle=particle, density=1E-3) for el in self.elements], weights=self.weights['mass_weight'], axis=0)*1E3*density
		
	def range(self, energy, particle='p', density=None):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		energy = np.asarray(energy, dtype=np.float64)
		
		dE = np.max(energy)/1E3
		E_min = min((np.min(energy), 1.0))
		E_grid = np.arange(E_min, np.max(energy)+dE, dE)

		S = self.S(E_grid, particle=particle, density=density)
		x = np.cumsum((1.0/S)*dE)
		return interp1d(np.log(E_grid), x, bounds_error=None, fill_value='extrapolate')(np.log(energy))
		
	def plot_mass_coeff(self, energy=None, **kwargs):
		"""Plot the mass-attenuation coefficient in the element

		Creates a plot of the mass-attenuation coefficient (in cm^2/g)
		as a function of photon energy in keV.

		Parameters
		----------
		energy : array_like, optional
			Energy grid on which to plot, replacing the default energy grid.
			Units are in keV.
		
		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> cm = ci.Compound('Fe')
		>>> cm.plot_mass_coeff()
		>>> cm.plot_mass_coeff(style='poster')

		"""

		if energy is None:
			energy, mu = self.mass_coeff['energy'], self.mass_coeff['mu']
			
		else:
			energy = np.asarray(energy, dtype=np.float64)
			mu = self.mu(energy)

		f,ax = _init_plot(**kwargs)

		ax.plot(energy, mu, label=r'$\mu/\rho$'+' ({})'.format(self.name))
		ax.set_xlabel('Photon Energy (keV)')
		ax.set_ylabel(r'Attenuation Coeff. (cm$^2$/g)')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend()
		
		return _draw_plot(f, ax, **kwargs)
		
	def plot_mass_coeff_en(self, energy=None, **kwargs):
		"""Plot the mass energy-absorption coefficient in the element

		Creates a plot of the mass energy-absorption coefficient (in cm^2/g)
		as a function of photon energy in keV.

		Parameters
		----------
		energy : array_like, optional
			Energy grid on which to plot, replacing the default energy grid.
			Units are in keV.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> cm = ci.Compound('Hf')
		
		Example plotting the mass-attenuation coefficient together with the mass
		energy-absorption coefficient, on the same axes.

		>>> f,ax = cm.plot_mass_coeff(return_plot=True)
		>>> cm.plot_mass_coeff_en(f=f, ax=ax)

		"""

		if energy is None:
			energy, mu = self.mass_coeff['energy'], self.mass_coeff['mu_en']
			
		else:
			energy = np.asarray(energy, dtype=np.float64)
			mu = self.mu_en(energy)

		f,ax = _init_plot(**kwargs)

		ax.plot(energy, mu, label=r'$\mu_{en}/\rho$'+' ({})'.format(self.name))
		ax.set_xlabel('Photon Energy (keV)')
		ax.set_ylabel(r'Attenuation Coeff. (cm$^2$/g)')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend()

		return _draw_plot(f, ax, **kwargs)
		
		
	def plot_S(self, particle='p', energy=None, **kwargs):
		"""Plot the stopping power in the element

		Creates a plot of the charged particle stopping power (in MeV/(mg/cm^2))
		in the element as a function of the incident ion energy (in MeV).

		Parameters
		----------
		particle : str
			Incident ion.  For light ions, options are 'p' (default), 'd', 't', 'a' for proton, 
			deuteron, triton and alpha, respectively.  Additionally, heavy ions can be
			specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'.

		energy : array_like, optional
			Energy grid on which to plot, replacing the default energy grid.
			Units are in MeV.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> el = ci.Element('He')
		>>> el.plot_S(particle='a')
		>>> el = ci.Element('Fe')
		>>> el.plot_S(particle='d')


		"""

		if energy is None:
			energy = 10.0**np.arange(-1.5, 2.8, 0.05)

		f,ax = _init_plot(**kwargs)
		ax.plot(energy, self.S(energy, particle=particle, density=1E-3), label='S ({})'.format(self.name))

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel(r'Stopping Power (MeV/(mg/cm$^2$))')
		ax.set_xscale('log')
		ax.legend()

		return _draw_plot(f, ax, **kwargs)
		
	def plot_range(self, particle='p', energy=None, density=None, **kwargs):
		"""Plot the charged particle range in the element

		Creates a plot of the charged particle range (in cm)
		in the element as a function of the incident ion energy (in MeV).

		Parameters
		----------
		particle : str
			Incident ion.  For light ions, options are 'p' (default), 'd', 't', 'a' for proton, 
			deuteron, triton and alpha, respectively.  Additionally, heavy ions can be
			specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'.

		energy : array_like, optional
			Energy grid on which to plot, replacing the default energy grid.
			Units are in MeV.

		density : float, optional
			Density of the element in g/cm^3.  Default behavior is to use
			`Element.density`.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> el = ci.Element('Ar')
		>>> el.plot_range()
		>>> el.plot_range(density=0.5)

		"""

		if energy is None:
			energy = 10.0**np.arange(-1.5, 2.8, 0.05)

		f,ax = _init_plot(**kwargs)
		ax.plot(energy, self.range(energy, particle=particle, density=density), label='Range ({})'.format(self.name))

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel('Range (cm)')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend()

		return _draw_plot(f, ax, **kwargs)