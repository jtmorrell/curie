from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .data import _get_connection
from .plotting import _init_plot, _draw_plot
from .isotope import Isotope

ELEMENTS = ['n','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
			'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
			'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
			'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',
			'Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy',
			'Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au',
			'Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U',
			'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db',
			'Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts']


class Element(object):
	"""Elemental data and properties

	The Element class provides useful data about the natural elements,
	such as mass, density, and isotopic composition.  Additionally, it contains
	functions for determining the interaction of radiation with the natural
	elements.  Principally it provides the mass-attenuation of photons, and 
	the stopping power/ranges of charged particles.
	
	Parameters
	----------
	element : str
		Symbol for the element, e.g. 'H', 'In', 'Zn', 'Fe'.  Case insensitive.
		Note that 'n' ("neutron") is not considered a valid element in this
		context, and will be interpreted as 'N' ("nitrogen").

	Attributes
	----------
	name : str
		Symbol for the element, in title-case. E.g. if input was 'fe', name 
		will be 'Fe'.

	Z : int
		Atomic number of the element.

	mass : float
		Molar mass of the natural element in atomic mass units (amu).

	isotopes : list
		List of isotopes with non-zero natural abundance.

	abundances : pd.DataFrame
		Natural abundances, in percent, for all isotopes found in nature. 
		Structure is a DataFrame with the columns 'isotope', 'abundance', 
		and 'unc_abundance'.

	density : float
		Density of the natural element in g/cm^3. The density is used in
		calculations of charged particle dEdx and photon attenuation, so
		you can assign a new density using `el.density = new_density` if
		needed, or using the `density` keyword in either of those functions.

	mass_coeff : pd.DataFrame
		Table of mass-attenuation coefficients as a function of photon
		energy, from the NIST XCOM database.  Energies are in keV, and
		mass-attenuation coefficients, or mu/rho, are given in cm^2/g.
		DataFrame columns are 'energy', 'mu' and 'mu_en' for the 
		mass-energy absorption coefficient.

	Examples
	--------
	>>> el = ci.Element('Fe')
	>>> print(el.mass)
	55.847
	>>> print(el.density)
	7.866

	"""

	def __init__(self, element):
		self.name = element.title()
		self.Z = ELEMENTS.index(self.name)

		df = pd.read_sql('SELECT * FROM weights WHERE Z={}'.format(self.Z), _get_connection('ziegler'))
		self.mass = df['amu'][0]
		self.density = df['density'][0]

		self.abundances = pd.read_sql('SELECT isotope, abundance, unc_abundance FROM chart WHERE Z={} AND abundance>0'.format(self.Z), _get_connection('decay'))
		if self.name=='Ta': ### Ta has a naturally occuring isomer
			self.abundances['isotope'] = map(lambda i:i.replace('180TA','180TAm1'), self.abundances['isotope'])
		self.isotopes = list(map(str, self.abundances['isotope']))

		self.mass_coeff = pd.read_sql('SELECT energy, mu, mu_en FROM mass_coeff WHERE Z={}'.format(self.Z), _get_connection('ziegler'))
		self._mc_interp, self._mc_en_interp = None, None
		self._A_p, self._A_he, self._I_p = None, None, None


	def __str__(self):
		return self.name

	def mu(self, energy):
		"""Mass-attenuation coefficient

		Interpolates the mass-attenuation coefficient, mu/rho,
		for the element along the input energy grid.

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
		>>> el = ci.Element('Hg')
		>>> print(el.mu(200))
		0.9456

		"""

		if self._mc_interp is None:
			self._mc_interp = interp1d(np.log(self.mass_coeff['energy']), np.log(self.mass_coeff['mu']), bounds_error=False, fill_value='extrapolate')
		return np.exp(self._mc_interp(np.log(energy)))

	def mu_en(self, energy):
		"""Mass energy-absorption coefficient

		Interpolates the mass-energy absorption coefficient, mu_en/rho,
		for the element along the input energy grid.

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
		>>> el = ci.Element('Hg')
		>>> print(el.mu_en(200))
		0.5661

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
			Thickness of slab of given element, in cm. 

		density : float, optional
			Density of the element in g/cm^3.  Default behavior is to
			use `Element.density`.

		Returns
		-------
		attenuation : numpy.ndarray
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


	def _parse_particle(self, particle):
		### Return Z, mass of incident particle
		if particle in ['p', 'd', 't', 'a']:
			return {'p':(1, 1.008), 'd':(1, 2.014), 't':(1, 3.016), 'a':(2, 4.002)}[particle]

		if particle=='P':
			print("WARNING: Assumed particle type P (phosphorus). If proton, use particle='p' (lower case).")
			return 15, 30.974

		### check if element or isotope
		if ''.join(re.findall('[A-Z]+', particle.upper())).title()==particle.title():
			el = Element(particle.title())
			return el.Z, el.mass

		else:
			ip = Isotope(particle)
			return ip.Z, ip.mass

	def _S_nucl(self, eng, z1, m1):
		RM = (m1+self.mass)*np.sqrt((z1**(2/3.0)+self.Z**(2/3.0)))
		ER = 32.53*self.mass*1E3*eng/(z1*self.Z*RM)

		return (0.5*np.log(1.0+ER)/(ER+0.10718+ER**0.37544))*8.462*z1*self.Z*m1/RM

	def _S_p(self, eng, M1=1.00727647):
		S = np.zeros(len(eng), dtype=np.float64) if eng.shape else np.array(0.0)
		E = 1E3*eng/M1

		if self._A_p is None:
			self._A_p = pd.read_sql('SELECT * FROM protons WHERE Z={}'.format(self.Z), _get_connection('ziegler')).to_numpy()[0,1:]
		A = self._A_p

		beta_sq = np.where(E>=1E3, 1.0-1.0/(1.0+E/931478.0)**2, 0.9)
		B0 = np.where(E>=1E3, np.log(A[6]*beta_sq/(1.0-beta_sq))-beta_sq, 0.0)
		Y = np.log(E[(E>=1E3)&(E<=5E4)])
		B0[np.where((E>=1E3)&(E<=5E4), B0, 0)!=0] -= A[7]+A[8]*Y+A[9]*Y**2+A[10]*Y**3+A[11]*Y**4

		S[E>=1E3] = (A[5]/beta_sq[E>=1E3])*B0[E>=1E3]

		S_low = A[1]*E[(E>=10)&(E<1E3)]**0.45
		S_high = (A[2]/E[(E>=10)&(E<1E3)])*np.log(1.0+(A[3]/E[(E>=10)&(E<1E3)])+A[4]*E[(E>=10)&(E<1E3)])

		S[(E>=10)&(E<1E3)] = S_low*S_high/(S_low+S_high)
		S[(E>0)&(E<10)] = A[0]*E[(E>0)&(E<10)]**0.5
		return S

	def _S_He(self, eng, M1=4.003):
		S = np.zeros(len(eng), dtype=np.float64) if eng.shape else np.array(0.0)
		E = eng*4.0015/M1
		E = np.where(E>=0.001, E, 0.001)

		if self._A_he is None:
			self._A_he = pd.read_sql('SELECT * FROM helium WHERE Z={}'.format(self.Z), _get_connection('ziegler')).to_numpy()[0,1:]
		A = self._A_he

		S_low = A[0]*(1E3*E[E<=10])**A[1]
		S_high = (A[2]/E[E<=10])*np.log(1.0+(A[3]/E[E<=10])+A[4]*E[E<=10])
		S[E<=10] = S_low*S_high/(S_low+S_high)

		Y = np.log(1.0/E[E>10])
		S[E>10] = np.exp(A[5]+A[6]*Y+A[7]*Y**2+A[8]*Y**3)
		return S

	def _S_elec(self, eng, z1, M1):
		S = np.zeros(len(eng), dtype=np.float64) if eng.shape else np.array(0.0)
		E_keV = 1E3*eng

		S[E_keV/M1<1000] = self._eff_Z_ratio(E_keV[E_keV/M1<1000], z1, M1)**2*self._S_p(eng[E_keV/M1<1000], M1)

		Y = E_keV[E_keV/M1>=1000]/M1
		beta_sq = 1.0-1.0/(1.0+Y/931478.0)**2
		FX = np.log(2E6*0.511003*beta_sq/(1.0-beta_sq))-beta_sq

		ZHY = 1.0-np.exp(-0.2*np.sqrt(Y)-0.0012*Y-0.00001443*Y**2)
		Z1EFF = self._eff_Z_ratio(E_keV[E_keV/M1>=1000], z1, M1)*ZHY

		if self._I_p is None:
			self._I_p = pd.read_sql('SELECT Ip_solid FROM ionization WHERE Z={}'.format(self.Z), _get_connection('ziegler'))['Ip_solid'][0]

		S[E_keV/M1>=1000] = 4E-1*np.pi*(1.9732857/137.03604)**2*Z1EFF**2*self.Z*(FX-np.log(self._I_p))/(0.511003*beta_sq)
		return S

	def _eff_Z_ratio(self, E_keV, z1, M1):
		if z1==1:
			return np.ones(len(eng))

		elif z1==2:
			Y = np.log(E_keV/M1)
			return z1*(1.0-np.exp(-0.7446-0.1429*Y-0.01562*Y**2+0.00267*Y**3-0.000001325*Y**8))

		elif z1==3:
			Y = E_keV/M1
			return z1*(1.0-np.exp(-0.7138-0.002797*Y-0.000001348*Y**2))

		BB = -0.886*np.sqrt(0.04*E_keV/M1)/z1**(2/3.0)
		return z1*(1.0-np.exp(BB-0.0378*np.sin(0.5*np.pi*BB))*(1.034-0.1777*np.exp(-0.08114*z1)))
		
	def S(self, energy, particle='p', density=None):
		"""Charged particle stopping power in matter

		Calculate the stopping power, S=-dE/dx, for a given ion as a 
		function of the ion energy in MeV.  Units of S are MeV/cm.  To return
		stopping power in units of MeV/(mg/cm^2), use option `density=1E-3`.

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
			Density of the element in g/cm^3.  Default behavior is to use
			`Element.density`.  To return stopping power in units of MeV/(mg/cm^2), i.e.
			the mass-stopping power, use `density=1E-3`.

		Returns
		-------
		stopping_power : numpy.ndarray
			Stopping power, S=-dE/dx, for a given ion as a function of the 
			ion energy in MeV.  Units of S are MeV/cm.

		Examples
		--------
		>>> el = ci.Element('La')
		>>> print(el.S(60.0))
		36.8687750516453
		>>> print(el.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)
		0.006371657662505643

		"""

		energy = np.asarray(energy, dtype=np.float64)
		if density is None:
			density = self.density

		Z_p, mass_p = self._parse_particle(particle)

		S = self._S_nucl(energy, Z_p, mass_p)
		if Z_p==1:
			S += self._S_p(energy, mass_p)
		elif Z_p==2:
			S += self._S_He(energy, mass_p)
		else:
			S += self._S_elec(energy, Z_p, mass_p)

		return (S*0.6022140857/self.mass)*1E3*density
		
	def range(self, energy, particle='p', density=None):
		"""Charged particle range in matter

		Calculates the charged particle range in the element, in cm.  Incident
		energy should be in MeV, and the particle type definition is identical
		to `Element.S()`.

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
			Density of the element in g/cm^3.  Default behavior is to use
			`Element.density`.
		
		Returns
		-------
		range : np.ndarray
			Charged particle range in the element, in cm.

		Examples
		--------
		>>> el = ci.Element('Fe')
		>>> print(el.range(60.0))
		0.5858151125192633
		>>> el = ci.Element('U')
		>>> print(el.range(60.0))
		0.3763111404628591

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
		>>> el = ci.Element('Fe')
		>>> el.plot_mass_coeff()
		>>> el.plot_mass_coeff(style='poster')

		"""

		if energy is None:
			energy, mu = self.mass_coeff['energy'], self.mass_coeff['mu']
			
		else:
			energy = np.asarray(energy, dtype=np.float64)
			mu = self.mu(energy)

		f,ax = _init_plot(**kwargs)

		ax.plot(energy, mu, label=r'$\mu/\rho$'+' ({0}, Z={1})'.format(self.name, self.Z))
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
		>>> el = ci.Element('Hf')
		
		Example plotting the mass-attenuation coefficient together with the mass
		energy-absorption coefficient, on the same axes.

		>>> f,ax = el.plot_mass_coeff(return_plot=True)
		>>> el.plot_mass_coeff_en(f=f, ax=ax)

		"""

		if energy is None:
			energy, mu = self.mass_coeff['energy'], self.mass_coeff['mu_en']
			
		else:
			energy = np.asarray(energy, dtype=np.float64)
			mu = self.mu_en(energy)

		f,ax = _init_plot(**kwargs)

		ax.plot(energy, mu, label=r'$\mu_{en}/\rho$'+' ({0}, Z={1})'.format(self.name, self.Z))
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
		ax.plot(energy, self.S(energy, particle=particle, density=1E-3), label='S ({0}, Z={1})'.format(self.name, self.Z))

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
		ax.plot(energy, self.range(energy, particle=particle, density=density), label='Range ({0}, Z={1})'.format(self.name, self.Z))

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel('Range (cm)')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend()

		return _draw_plot(f, ax, **kwargs)