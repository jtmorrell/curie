from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import json
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .data import _get_connection
from .plotting import _init_plot, _draw_plot
from .element import ELEMENTS, Element

COMPOUND_LIST = list(map(str, pd.read_sql('SELECT compound FROM compounds', _get_connection('ziegler'))['compound']))

class Compound(object):
	"""Data and properties of atomic compounds

	The compound class provides the same functionality as the
	element class, but for compounds of atomic elements rather
	than the individual atomic elements.  The compound is described
	by a set of elements, a set of weights for each element, and a 
	density.  

	The weights can be either given as atom-weights, e.g.
	in H2O the atom weights are 0.67 for H and 0.33 for O, or as 
	mass-weights, e.g. brass is 0.33 Zn by weight and 0.67 Cu by
	weight.  Some preset compounds are available; their names can be
	found by printing `ci.COMPOUND_LIST`.
	
	Parameters
	----------
	compound : str
		The name of the compound.  If the compound is in `ci.COMPOUND_LIST`,
		the weights and density will take preset values, if not given.
		If the name of the compound is given in chemical notation, e.g. NaCl
		or H2O2, the atom-weights can be inferred if not given explicitly.  Note
		that full chemical notation is not supported, so Ca3(PO4)2 must be written
		as Ca3P2O8.  Decimal weights are supported, e.g. C0.5O is equivalent to CO2.

	weights : dict, str or pd.DataFrame, optional
		The weights of each element in the compound.  Multiple formats are supported.
		If weights is a dict, it must be formatted as {'el1':wt1, 'el2':wt2}, where
		atom-weights are positive, and mass-weights are negative. 

		If weights is a pandas DataFrame, it must contain an 'element' column, 
		and one of 'weight', 'atom_weight', or 'mass_weight'.  If 'weight' is the column 
		given, the convention of positive atom-weights and negative mass-weights is followed.  

		If weights is a str, it can either be formatted as 'el1:wt1, el2:wt2', or it can be 
		a path to a .csv, .json or .db file.  These files must contain the same information 
		as the DataFrame option, and can contain weights for multiple compounds, if 'compound' 
		is one of the columns/keys.
		
		If a .json file, it must follow the 'records' formatting convention (see pandas docs).

	density : float, optional
		Density of the compound in g/cm^3.  The density is required for the cm.attenuation(),
		cm.S(), cm.range() and cm.plot_range() functions, but is an optional argument in each
		of those functions if not provided at construction.  Can also be specified by using
		a 'density' column/key in the file/DataFrame for weights.

	Attributes
	----------
	name : str
		The name of the compound.

	weights : pd.DataFrame
		The weights for each element in the compound.  DataFrame columns are
		'element', 'Z', 'mass_weight', 'atom_weight'.

	density : float
		Density of the compound in g/cm^3. The density is used in
		calculations of charged particle dEdx and photon attenuation, so
		if the density was not explicitly given at construction,
		you can assign a new density using `cm.density = new_density` if
		needed, or using the `density` keyword in either of those functions.

	elements : list of ci.Element
		Elements in the compound.

	mass_coeff : pd.DataFrame
		Table of mass-attenuation coefficients as a function of photon
		energy, from the NIST XCOM database.  Energies are in keV, and
		mass-attenuation coefficients, or mu/rho, are given in cm^2/g.
		DataFrame columns are 'energy', 'mu' and 'mu_en' for the 
		mass-energy absorption coefficient.

	Examples
	--------
	>>> print('Silicone' in ci.COMPOUND_LIST)
	True
	>>> cm = ci.Compound('Silicone') # preset compound
	>>> print(list(map(str, cm.elements)))
	['H', 'C', 'O', 'Si']
	>>> cm = ci.Compound('H2O', density=1.0)
	print(cm.weights)
	  element  Z  atom_weight  mass_weight
	0       H  1     0.666667     0.111907
	1       O  8     0.333333     0.888093
	>>> cm = ci.Compound('Brass', weights={'Zn':-33,'Cu':-66})
	>>> print(cm.weights)
	  element   Z  atom_weight  mass_weight
	0      Zn  30     0.327041     0.333333
	1      Cu  29     0.672959     0.666667
	>>> cm.saveas('brass.csv')

	"""

	def __init__(self, compound, weights=None, density=None):
		self.name = compound
		self.density = None

		if compound in COMPOUND_LIST:
			df = pd.read_sql('SELECT * FROM compounds WHERE compound="{}"'.format(compound), _get_connection('ziegler'))
			self.density = df['density'][0]

			if weights is None:
				wts = df['weights'][0].split(',')
				elements = [str(i.split(':')[0]) for i in wts]
				atom_weights = np.array([float(i.split(':')[1]) for i in wts])
				self._set_weights(elements, atom_weights=atom_weights)
			
		elif weights is None:
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
				weights = pd.DataFrame({'element':elements, 'weight':wts})

			elif type(weights)==str:
				if weights.endswith('.json'):
					weights = pd.read_json(weights, orient='records').fillna(method='ffill')
					weights.columns = map(str.lower, map(str, weights.columns))
					if 'compound' in weights.columns:
						weights = weights[weights['compound']==self.name]

				elif weights.endswith('.csv'):
					weights = pd.read_csv(weights, header=0).fillna(method='ffill')
					weights.columns = map(str.lower, map(str, weights.columns))
					if 'compound' in weights.columns:
						weights = weights[weights['compound']==self.name]

				elif weights.endswith('.db'):
					weights = pd.read_sql('SELECT * FROM compounds WHERE compound={}'.format(self.name), _get_connection(weights))
					weights.columns = map(str.lower, map(str, weights.columns))
					if 'compound' in weights.columns:
						weights = weights[weights['compound']==self.name]

				else:
					elements = [str(i.split(':')[0]).strip() for i in weights.split(',')]
					wts = np.array([float(i.split(':')[1].strip()) for i in weights.split(',')])

					if wts[0]>0:
						self._set_weights(elements, atom_weights=wts)
					else:
						self._set_weights(elements, mass_weights=np.abs(wts))

			if type(weights)==pd.DataFrame:
				weights.columns = map(str.lower, map(str, weights.columns))
				if 'density' in weights.columns:
					self.density = weights['density'].iloc[0]

				cols = ['element', 'Z', 'atom_weight', 'mass_weight']
				if all([i in weights.columns for i in cols]):
					self.weights = weights[cols]

				elif 'atom_weight' in weights.columns:
					self._set_weights(list(weights['element']), atom_weights=weights['atom_weight'].to_numpy())

				elif 'mass_weight' in weights.columns:
					self._set_weights(list(weights['element']), mass_weights=weights['mass_weight'].to_numpy())

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


	def __str__(self):
		return self.name

	def saveas(self, filename, replace=False):
		"""Save the compound definition to a file

		The weights and density of the compound can be saved to one of
		the following file formats: .csv, .json, .db.  If the file exists,
		the data will be appended, unless `replace=True`, in which case
		the file will be replaced.  If a definition for the compound exists
		in the file already, it will be replaced.

		Parameters
		----------
		filename : str
			Filename where the compound will be saved.  Available formats
			are .csv, .json and .db.

		replace : bool, optional
			If `True`, replace the file if it exists.  Default `False`, which
			appends the data to the file.

		Examples
		--------
		>>> cm = ci.Compound('Brass', weights={'Zn':-33,'Cu':-66})
		>>> cm.saveas('brass.csv')
		>>> cm = ci.Compound('Water', weights={'H':2, 'O':1}, density=1.0)
		>>> cm.saveas('water.json')

		"""
		wts = self.weights.copy()
		wts['compound'] = self.name
		if self.density is not None:
			wts['density'] = self.density

		if filename.endswith('.csv'):
			if os.path.exists(filename) and not replace:
				df = pd.read_csv(filename, header=0)
				df = df[df['compound']!=self.name]
				df = pd.concat([df, wts])
				df.to_csv(filename, index=False)
			else:
				wts.to_csv(filename, index=False)

		if filename.endswith('.db'):
			if os.path.exists(filename) and not replace:
				con = _get_connection(filename)
				df = pd.read_sql('SELECT * FROM weights', con)
				df = df[df['compound']!=self.name]
				df = pd.concat([df, wts])
				df.to_sql('weights', con, if_exists='replace', index=False)
			else:
				wts.to_sql('weights', _get_connection(filename), if_exists='replace', index=False)

		if filename.endswith('.json'):
			if os.path.exists(filename) and not replace:
				df = pd.read_json(filename, orient='records')
				df = df[df['compound']!=self.name][wts.columns]
				df = pd.concat([df, wts])
				json.dump(json.loads(df.to_json(orient='records')), open(filename, 'w'), indent=4)
			else:
				json.dump(json.loads(wts.to_json(orient='records')), open(filename, 'w'), indent=4)

	def mu(self, energy):
		"""Mass-attenuation coefficient

		Interpolates the mass-attenuation coefficient, mu/rho,
		for the compound along the input energy grid.

		Parameters
		----------
		energy : array_like
			The incident photon energy, in keV.

		Returns
		-------
		mu : np.ndarray
			Mass attenuation coefficient, mu/rho, in cm^2/g.

		Examples
		--------
		>>> cm = ci.Compound('H2O')
		>>> print(cm.mu(200))
		0.13703928393005832

		"""

		if self._mc_interp is None:
			self._mc_interp = interp1d(np.log(self.mass_coeff['energy']), np.log(self.mass_coeff['mu']), bounds_error=False, fill_value='extrapolate')
		return np.exp(self._mc_interp(np.log(energy)))

	def mu_en(self, energy):
		"""Mass energy-absorption coefficient

		Interpolates the mass-energy absorption coefficient, mu_en/rho,
		for the compound along the input energy grid.

		Parameters
		----------
		energy : array_like
			The incident photon energy, in keV.

		Returns
		-------
		mu_en : np.ndarray
			Mass energy absorption coefficient, mu_en/rho, in cm^2/g.

		Examples
		--------
		>>> cm = ci.Compound('H2O')
		>>> print(cm.mu_en(200))
		0.029671598667776862

		"""

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
			Thickness of slab of given compound, in cm. 

		density : float, optional
			Density of the compound in g/cm^3.  Default behavior is to
			use `Compound.density`, which must be supplied at construction.

		Returns
		-------
		attenuation : numpy.ndarray
			The slab attenuation factor as an absolute number (i.e. from 0 to 1).
			E.g. if the incident intensity is I_0, the transmitted intensity I(x) 
			is I_0 times the attenuation factor.

		Examples
		--------
		>>> cm = ci.Compound('SS_316') # preset compound for 316 Stainless
		>>> print(cm.attenuation(511, x=0.3))
		0.8199829388434694
		>>> print(cm.attenuation(300, x=1.0, density=5.0))
		0.5752140388004373

		"""

		energy = np.asarray(energy, dtype=np.float64)
		x = np.asarray(x, dtype=np.float64)
		if density is None:
			density = self.density

		return np.exp(-self.mu(energy)*x*density)
		
	def S(self, energy, particle='p', density=None):
		"""Charged particle stopping power in matter

		Calculate the stopping power, S=-dE/dx, for a given ion as a 
		function of the ion energy in MeV.  Units of S are MeV/cm.  To return
		stopping power in units of MeV/(mg/cm^2), use option `density=1E-3`.
		The stopping power is calculated using the Element.S() methods for
		each element in cm.elements, added using Bragg's rule.

		Parameters
		----------
		energy : array_like
			Incident ion energy in MeV.

		particle : str, optional
			Incident ion.  For light ions, options are 'p' (default), 'd', 't', 'a' for proton, 
			deuteron, triton and alpha, respectively.  Additionally, heavy ions can be
			specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'. For
			light ions, the charge state is assumed to be fully stripped. For heavy ions
			the charge state is handled by a Bohr/Northcliffe parameterization consistent
			with the Anderson-Ziegler formalism.

		density : float, optional
			Density of the compound in g/cm^3.  Default behavior is to use
			`Compound.density`.  To return stopping power in units of MeV/(mg/cm^2), i.e.
			the mass-stopping power, use `density=1E-3`.

		Returns
		-------
		stopping_power : numpy.ndarray
			Stopping power, S=-dE/dx, for a given ion as a function of the 
			ion energy in MeV.  Units of S are MeV/cm.

		Examples
		--------
		>>> cm = ci.Compound('SrCO3', density=3.5)
		>>> print(cm.S(60.0))
		27.196387031247834
		>>> print(cm.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)
		0.008307827781861116

		"""

		energy = np.asarray(energy, dtype=np.float64)
		if density is None:
			density = self.density

		return np.average([el.S(energy, particle=particle, density=1E-3) for el in self.elements], weights=self.weights['mass_weight'], axis=0)*1E3*density
		
	def range(self, energy, particle='p', density=None):
		"""Charged particle range in matter

		Calculates the charged particle range in the compound, in cm.  Incident
		energy should be in MeV, and the particle type definition is identical
		to `Compound.S()`.

		Parameters
		----------
		energy : array_like
			Incident ion energy in MeV.

		particle : str, optional
			Incident ion.  For light ions, options are 'p' (default), 'd', 't', 'a' for proton, 
			deuteron, triton and alpha, respectively.  Additionally, heavy ions can be
			specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'. For
			light ions, the charge state is assumed to be fully stripped. For heavy ions
			the charge state is handled by a Bohr/Northcliffe parameterization consistent
			with the Anderson-Ziegler formalism.

		density : float, optional
			Density of the compound in g/cm^3.  Default behavior is to use
			`Compound.density`, which must be supplied at construction.
		
		Returns
		-------
		range : np.ndarray
			Charged particle range in the compound, in cm.

		Examples
		--------
		>>> cm = ci.Compound('Fe') # same behavior as element
		>>> print(cm.range(60.0))
		0.5858151125192633
		>>> cm = ci.Compound('SS_316') # preset compound
		>>> print(cm.range(60.0))
		0.5799450918147814

		"""

		energy = np.asarray(energy, dtype=np.float64)
		
		dE = np.max(energy)/1E3
		E_min = min((np.min(energy), 1.0))
		E_grid = np.arange(E_min, np.max(energy)+dE, dE)

		S = self.S(E_grid, particle=particle, density=density)
		x = np.cumsum((1.0/S)*dE)
		return interp1d(np.log(E_grid), x, bounds_error=None, fill_value='extrapolate')(np.log(energy))
		
	def plot_mass_coeff(self, energy=None, **kwargs):
		"""Plot the mass-attenuation coefficient in the compound

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
		>>> cm = ci.Compound('H2O')
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
		"""Plot the mass energy-absorption coefficient in the compound

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
		>>> cm = ci.Compound('Silicone') # preset compound
		
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
		"""Plot the stopping power in the compound

		Creates a plot of the charged particle stopping power (in MeV/(mg/cm^2))
		in the compound as a function of the incident ion energy (in MeV).

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
		>>> cm = ci.Compound('He') # same as element
		>>> cm.plot_S(particle='a')
		>>> cm = ci.Compound('Kapton')
		>>> cm.plot_S(particle='d')

		"""

		if energy is None:
			energy = 10.0**np.arange(-1.5, 2.8, 0.05)

		f,ax = _init_plot(**kwargs)
		ax.plot(energy, self.S(energy, particle=particle, density=1E-3), label=r'$-\frac{dE}{dx}$ ('+self.name+')')

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel(r'Stopping Power (MeV/(mg/cm$^2$))')
		ax.set_xscale('log')
		ax.legend()

		return _draw_plot(f, ax, **kwargs)
		
	def plot_range(self, particle='p', energy=None, density=None, **kwargs):
		"""Plot the charged particle range in the compound

		Creates a plot of the charged particle range (in cm)
		in the compound as a function of the incident ion energy (in MeV).

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
			Density of the compound in g/cm^3.  Default behavior is to use
			`Compound.density`.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> cm = ci.Compound('Bronze', weights={'Cu':-80, 'Sn':-20}, density=8.9)
		>>> f,ax = cm.plot_range(return_plot=True)
		>>> cm.plot_range(particle='d', f=f, ax=ax)

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