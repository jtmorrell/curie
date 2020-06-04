from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd

from .data import _get_connection

ELEMENTS = ['n','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
			'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co',
			'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
			'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',
			'Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy',
			'Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au',
			'Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U',
			'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db',
			'Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts']

class Isotope(object):
	"""Retrieve isotopic structure and decay data

	The Isotope class provides isotope specific nuclear data, such as masses,
	abundances, half-lives, decay data and fission yields.  The main data souces 
	are NuDat 2.0, ENDF/B-VII.0 and the nuclear wallet cards.  Where conflicts
	were found preference was given to NuDat, then ENDF, then wallet cards.
	
	Parameters
	----------
	isotope : str
		Name of the isotope/isomer.  For element El of mass AAA, and (optionally)
		isomeric state m# (where # is an integer starting with 1 for the first isomer)
		or g for ground state, the name must be formatted as either AAAELm# or El-AAAm#
		or EL-AAAm#. If the isomeric state isn't specified, the ground state is assumed,
		and if just m is given, the first isomer is assumed.

	Attributes
	----------
	element : str
		Elemental symbol, e.g. Xe, Ar, Kr

	A : int
		Mass number A of isotope

	isomer : str
		Isomeric state, e.g. g, m1, m2

	name : str
		Formatted isotope name

	E_level : float
		Energy level of the state, in MeV

	J_pi : str
		Spin and partiy of the state

	Z : int
		Atomic number Z of the isotope

	N : int
		Neutron number N of the isotope

	dc : float
		Isotope decay constant in units of 1/seconds

	stable : bool
		`True` if isotope is stable

	mass : float
		Atomic mass of the isotope in amu

	abundance : float
		Natural abundance in percent.

	unc_abundance : float
		Uncertainty in natural abundance in percent.

	Delta : float
		Mass excess of the isotope in MeV

	decay_products : dict
		Dictionary of decay products and their absolute branching ratios as `{'istp':BR}`

	TeX : str
		LaTeX formatted isotope name

	Examples
	--------
	>>> ip = ci.Isotope('115INm')
	>>> ip = ci.Isotope('Co-60')
	>>> ip = ci.Isotope('58NI')
	>>> print(ip.abundance)
	68.077
	
	>>> ip = ci.Isotope('135CEm')
	>>> print(ip.dc)
	0.03465735902799726

	>>> ip = ci.Isotope('235U')
	>>> print(ip.half_life('My'))
	703.798767967


	"""

	def __init__(self, isotope):
		self._parse_itp_name(isotope)

		df = pd.read_sql('SELECT * FROM chart WHERE name="{}"'.format(self.name), _get_connection('decay'))
		self.E_level = float(df['E_level'][0])
		self.J_pi = str(df['J_pi'][0])
		self.stable = bool(df['stable'][0])
		if self.stable:
			self.dc = 0.0
			self.decay_products = {}
		self.mass = float(df['amu'][0])
		if df['abundance'][0] is not None:
			self.abundance = float(df['abundance'][0])
		else:
			self.abundance = 0.0
		if df['Delta'][0] is not None:
			self.Delta = float(df['Delta'][0])
		else:
			self.Delta = None

		if self.abundance>0.0:
			self.unc_abundance = float(df['unc_abundance'][0])
		else:
			self.unc_abundance = 0.0


		self._SFY = None

		if not self.stable:
			self._t_half = float(df['half_life'][0])
			self._unc_t_half = float(df['unc_half_life'][0])
			self.dc = np.log(2.0)/self._t_half

			self.decay_products = {}
			for mode in str(df['decay_mode'][0]).split(','):
				if len(mode.split(':'))!=3:
					continue
				dcy, itp, br = tuple(mode.split(':'))
				br = float(br)
				if itp=='SFY':
					for n,y in self.get_SFY().iterrows():
						self.decay_products[str(y['daughter'])] = y['yield']*br
				else:
					self.decay_products[str(itp)] = br


	def _parse_itp_name(self, istp):
		if istp=='1n' or istp=='1ng' or istp=='n':
			self.element, self.A, self.isomer = 'n', 1, 'g'
			self.name = '1ng'
			self._short_name = '1n'
			
		elif '-' in istp:
			el, A = tuple(istp.split('-'))
			if 'm' in A:
				A, m = tuple(A.split('m'))
				self.isomer = 'm'+m
				if self.isomer=='m':
					self.isomer = 'm1'

			elif 'g' in A:
				A = A[:-1]
				self.isomer = 'g'
			else:
				self.isomer = 'g'

			self.A = int(A)
			self.element = el.title()

		else:
			self.element = ''.join(re.findall('[A-Z]+', istp)).title()
			if istp.startswith('nat'):
				self.A = 'nat'
			else:
				self.A = int(istp.split(self.element.upper())[0])

			self.isomer = istp.split(self.element.upper())[1]
		
		if self.isomer=='' and type(self.A)==int:
			self.isomer = 'g'
		if self.isomer=='m':
			self.isomer = 'm1'

		if self.element!='n':
			self.name = str(self.A)+self.element.upper()+self.isomer
			self._short_name = str(self.A)+self.element.upper()

		self.Z = ELEMENTS.index(self.element)
		if type(self.A)==int:
			self.N = self.A-self.Z

		self.TeX = r'$^{'+str(self.A)+(self.isomer if self.isomer!='g' else '')+r'}$'+self.element


	def __str__(self):
		return self.name

	def half_life(self, units='s', unc=False):
		"""Half-life of isotope

		Returns isotope half-life, in specified units, with or without uncertainty.

		Parameters
		----------
		units : str, optional
			Units for which to return half-life. Options are 
			ns, us, ms, s, m, h, d, y, ky, My, Gy

		unc : bool, optional
			If `True`, uncertainty in half life (in specified units) will also be returned

		Returns
		-------
		half_life : float or 2-tuple of floats
			Half-life in specified units, with uncertainty only if `unc=True`

		Examples
		--------
		>>> ip = ci.Isotope('67CU')
		>>> print(ip.half_life('d'))
		2.57625
		>>> print(ip.half_life('h', True))
		(61.83, 0.12)

		"""

		if units in ['hr','min','yr','sec']:
			units = {'hr':'h','min':'m','yr':'y','sec':'s'}[units]

		half_conv = {'ns':1E-9, 'us':1E-6, 'ms':1E-3,
					's':1.0, 'm':60.0, 'h':3600.0,
					'd':86400.0, 'y':31557.6E3, 'ky':31557.6E6,
					'My':31557.6E9, 'Gy':31557.6E12}[units]

		if self.stable:
			return (np.inf,0.0) if unc else np.inf

		if unc:
			return self._t_half/half_conv, self._unc_t_half/half_conv

		return self._t_half/half_conv

	def decay_const(self, units='s', unc=False):
		"""Decay constant of isotope

		Returns isotope decay constant, in specified units, with or without uncertainty

		Parameters
		----------
		units : str, optional
			Units for which to return decay constant. Options are 
			ns, us, ms, s, m, h, d, y, ky, My, Gy
			Actual value will be given in units of inverse time

		unc : str, optional
			If `True`, uncertainty in decay constant (in specified units) will also be returned

		Returns
		-------
		decay_const : float or 2-tuple of floats
			Decay constant in specified units, with uncertainty only if `unc=True`

		Examples
		--------
		>>> ip = ci.Isotope('221AT')
		>>> print(ip.decay_const())
		0.0050228056562314875
		>>> print(ip.decay_const('m', True))
		(0.3013683393738893, 0.026205942554251245)

		"""

		if self.stable:
			return (0.0, 0.0) if unc else 0.0

		if unc:
			t_half, unc_t_half = self.half_life(units, True)
			dc = np.log(2.0)/t_half
			return dc, dc*(unc_t_half/t_half)

		return np.log(2.0)/self.half_life(units)

	def optimum_units(self):
		"""Appropriate units for half-life 

		Returns
		-------
		units : str
			Units that will represent half-life as a number greater than 1, but with as few digits as possible.

		Examples
		--------
		>>> ip = ci.Isotope('226RA')
		>>> print(ip.half_life())
		50492200000.0
		>>> print(ip.optimum_units())
		y
		>>> print(ip.half_life(ip.optimum_units()))
		1600.00126752

		"""

		opt = ['ns']
		for units in ['us','ms','s','m','h','d','y']:
			if self.half_life(units)>1.0:
				opt.append(units)
		return opt[-1]

	def get_SFY(self):
		"""Spontaneous fission yields

		Returns the absolute spontaneous fission yields (independent) for a given nuclide,
		where available from ENDF.
		
		Returns
		-------
		sfy : pd.DataFrame
			Tabular spontaneous fission yields, with keys 'daughter', 'yield' and 'unc_yield'

		Examples
		--------
		>>> ip = ci.Isotope('238U')
		>>> print(ip.get_SFY())

		"""

		if self._SFY is None:
			self._SFY = pd.read_sql('SELECT daughter, yield, unc_yield FROM SFY WHERE parent="{}"'.format(self.name), _get_connection('decay'))
		return self._SFY

	def get_NFY(self, E):
		"""Neutron induced fission yields

		Returns the absolute neutron induced fission yields (independent) for a given nuclide,
		where available from ENDF.

		Parameters
		----------
		E : float
			Incident neutron energy, in eV. Available energies are 0.0253, 5E5, 2E6 and 14E6.

		Returns
		-------
		nfy : pd.DataFrame
			Tabular neutron induced fission yields, with keys 'daughter', 'yield' and 'unc_yield'

		Examples
		--------
		>>> ip = ci.Isotope('235U')
		>>> print(ip.get_NFY(E=0.0253))

		"""

		return pd.read_sql('SELECT daughter, yield, unc_yield FROM NFY WHERE parent="{0}" AND energy={1}'.format(self.name, E), _get_connection('decay'))


	def gammas(self, I_lim=None, E_lim=None, xrays=False, dE_511=0.0, istp_col=False):
		"""Gamma-rays emitted by the decay of the isotope

		Returns a DataFrame of gamma-ray energies, intensities and intensity-uncertainties
		based on an optional set of selection critieria.

		Parameters
		----------
		I_lim : float or 2-tuple of floats, optional
			Limits on the intensity of the gamma-rays, in percent.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		E_lim : float or 2-tuple of floats, optional
			Limits on the energy of the gamma-rays, in keV.  If a single float is given,
			this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		xrays : bool, optional
			If `True`, retrieved data will include x-rays.  Default `False`.

		dE_511 : float, optional
			Filter out gamma-rays that are within `dE_511` keV of the annihilation peak.
			Default 0.0.

		Returns
		-------
		gammas : pd.DataFrame
			Tabular gamma-ray data, with keys 'energy', 'intensity' and 'unc_intensity'. Units
			of energy are in keV and units of intensity are in percent.

		Examples
		--------
		>>> ip = ci.Isotope('Co-60')
		>>> print(ip.gammas(I_lim=1.0))
		     energy  intensity  unc_intensity
		0  1173.228    99.8500         0.0300
		1  1332.492    99.9826         0.0006

		>>> ip = ci.Isotope('64CU')
		>>> print(ip.gammas())
			energy  intensity  unc_intensity
		0   511.00     35.200          0.400
		1  1345.77      0.475          0.011
		>>> print(ip.gammas(xrays=True, dE_511=1.0))
			 energy  intensity  unc_intensity
		0     0.850      0.489          0.024
		1     7.461      4.740          0.240
		2     7.478      9.300          0.400
		3     8.265      1.120          0.050
		4     8.265      0.580          0.030
		5  1345.770      0.475          0.011

		"""

		ip, im = str(self.A)+self.element.upper(), self.isomer

		if xrays:
			df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM gammas WHERE isotope="{0}" AND isomer="{1}"'.format(ip, im), _get_connection('decay'))
		else:
			df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM gammas WHERE isotope="{0}" AND isomer="{1}" AND notes NOT LIKE "%XR%"'.format(ip, im), _get_connection('decay'))

		df = df[np.abs(df['energy']-511.0)>=dE_511]

		if I_lim is not None:
			if type(I_lim)==float or type(I_lim)==int:
				df = df[df['intensity']>=I_lim]
			else:
				df = df[(df['intensity']>=I_lim[0])&(df['intensity']<=I_lim[1])]

		if E_lim is not None:
			if type(E_lim)==float or type(E_lim)==int:
				df = df[df['energy']>=E_lim]
			else:
				df = df[(df['energy']>=E_lim[0])&(df['energy']<=E_lim[1])]

		if istp_col:
			df['isotope'] = self.name

		return df.sort_values(by=['energy']).reset_index(drop=True)

	def electrons(self, I_lim=None, E_lim=None, CE_only=False, Auger_only=False):
		"""Electrons (conversion and Auger) emitted by the decay of the isotope

		Returns a DataFrame of electron energies, intensities and intensity-uncertainties
		based on an optional set of selection critieria.

		Parameters
		----------
		I_lim : float or 2-tuple of floats, optional
			Limits on the intensity of the electrons, in percent.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		E_lim : float or 2-tuple of floats, optional
			Limits on the energies of the electrons, in keV.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		CE_only : bool
			If `True`, only conversion electrons will be returned. Default `False`

		Auger_only : bool
			If `True`, only Auger electrons will be returned. Default `False`.

		Returns
		-------
		electrons : pd.DataFrame
			Tabular electron data, with keywords 'energy', 'intensity', 'unc_intensity'. Units
			of energy are in keV and units of intensity are in percent.

		Examples
		--------
		>>> ip = ci.Isotope('Pt-193m')
		>>> print(ip.electrons(I_lim=5.0, E_lim=(10.0, 130.0)))
		    energy  intensity  unc_intensity
		0   11.912       17.1          0.855
		1   57.110       15.5          0.775
		2  121.620       60.0          3.000

		"""

		ip, im = str(self.A)+self.element.upper(), self.isomer

		if CE_only:
			df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM electrons WHERE isotope="{0}" AND isomer="{1}" AND notes LIKE "%CE%"'.format(ip, im), _get_connection('decay'))
		elif Auger_only:
			df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM electrons WHERE isotope="{0}" AND isomer="{1}" AND notes LIKE "%Auger%"'.format(ip, im), _get_connection('decay'))
		else:
			df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM electrons WHERE isotope="{0}" AND isomer="{1}"'.format(ip, im), _get_connection('decay'))

		if I_lim is not None:
			if type(I_lim)==float or type(I_lim)==int:
				df = df[df['intensity']>=I_lim]
			else:
				df = df[(df['intensity']>=I_lim[0])&(df['intensity']<=I_lim[1])]

		if E_lim is not None:
			if type(E_lim)==float or type(E_lim)==int:
				df = df[df['energy']>=E_lim]
			else:
				df = df[(df['energy']>=E_lim[0])&(df['energy']<=E_lim[1])]

		return df.sort_values(by=['energy']).reset_index(drop=True)

	def beta_minus(self, I_lim=None, Endpoint_lim=None):
		"""Electrons from beta-minus decay

		Returns a DataFrame of beta-minus mean and enpoint energies, intensities and
		intensity-uncertainties based on an optional set of selection critieria.

		Parameters
		----------
		I_lim : float or 2-tuple of floats, optional
			Limits on the intensity of the beta-minus decay, in percent.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		Endpoint_lim : float or 2-tuple of floats, optional
			Limits on the endpoint energy of the beta-minus decay, in keV.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		Returns
		-------
		beta_minus : pd.DataFrame
			Tabular beta-minus data, with keywords 'mean_energy', 'endpoint_energy', 'intensity',
			'unc_intensity'. Units of energy are in keV and units of intensity are in percent.

		Examples
		--------
		>>> ip = ci.Isotope('35S')
		>>> print(ip.beta_minus())
		   mean_energy  endpoint_energy  intensity  unc_intensity
		0       48.758           167.33      100.0            5.0

		"""

		ip, im = str(self.A)+self.element.upper(), self.isomer

		df = pd.read_sql('SELECT mean_energy, endpoint_energy, intensity, unc_intensity FROM beta_minus WHERE isotope="{0}" AND isomer="{1}"'.format(ip, im), _get_connection('decay'))

		if I_lim is not None:
			if type(I_lim)==float or type(I_lim)==int:
				df = df[df['intensity']>=I_lim]
			else:
				df = df[(df['intensity']>=I_lim[0])&(df['intensity']<=I_lim[1])]

		if Endpoint_lim is not None:
			if type(Endpoint_lim)==float or type(Endpoint_lim)==int:
				df = df[df['endpoint_energy']>=Endpoint_lim]
			else:
				df = df[(df['endpoint_energy']>=Endpoint_lim[0])&(df['endpoint_energy']<=Endpoint_lim[1])]

		return df.sort_values(by=['endpoint_energy']).reset_index(drop=True)

	def beta_plus(self, I_lim=None, Endpoint_lim=None):
		"""Positrons from beta-plus decay

		Returns a DataFrame of beta-plus mean and enpoint energies, intensities and
		intensity-uncertainties based on an optional set of selection critieria.

		Parameters
		----------
		I_lim : float or 2-tuple of floats, optional
			Limits on the intensity of the beta-plus decay, in percent.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		Endpoint_lim : float or 2-tuple of floats, optional
			Limits on the endpoint energy of the beta-plus decay, in keV.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		Returns
		-------
		beta_plus : pd.DataFrame
			Tabular beta-plus data, with keywords 'mean_energy', 'endpoint_energy', 'intensity',
			'unc_intensity'. Units of energy are in keV and units of intensity are in percent.

		Examples
		--------
		>>> ip = ci.Isotope('18F')
		>>> print(ip.beta_plus())
		   mean_energy  endpoint_energy  intensity  unc_intensity
		0        249.8            633.5      96.73           0.04

		"""

		ip, im = str(self.A)+self.element.upper(), self.isomer

		df = pd.read_sql('SELECT mean_energy, endpoint_energy, intensity, unc_intensity FROM beta_plus WHERE isotope="{0}" AND isomer="{1}"'.format(ip, im), _get_connection('decay'))

		if I_lim is not None:
			if type(I_lim)==float or type(I_lim)==int:
				df = df[df['intensity']>=I_lim]
			else:
				df = df[(df['intensity']>=I_lim[0])&(df['intensity']<=I_lim[1])]

		if Endpoint_lim is not None:
			if type(Endpoint_lim)==float or type(Endpoint_lim)==int:
				df = df[df['endpoint_energy']>=Endpoint_lim]
			else:
				df = df[(df['endpoint_energy']>=Endpoint_lim[0])&(df['endpoint_energy']<=Endpoint_lim[1])]

		return df.sort_values(by=['endpoint_energy']).reset_index(drop=True)

	def alphas(self, I_lim=None, E_lim=None):
		"""Alpha-particles emitted by the decay of the isotope

		Returns a DataFrame of alpha-particle energies, intensities and intensity-uncertainties
		based on an optional set of selection critieria.

		Parameters
		----------
		I_lim : float or 2-tuple of floats, optional
			Limits on the intensity of the alphas, in percent.  If a single float
			is given, this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		E_lim : float or 2-tuple of floats, optional
			Limits on the energy of the alphas, in keV.  If a single float is given,
			this is a lower bound, if a 2-tuple then (upper, lower) bounds.

		Returns
		-------
		alphas : pd.DataFrame
			Tabular alpha emission data, with keys 'energy', 'intensity' and 'unc_intensity'. Units
			of energy are in keV and units of intensity are in percent.

		Examples
		--------
		>>> ip = ci.Isotope('210PO')
		>>> print(ip.alphas(I_lim=1.0))
		    energy  intensity  unc_intensity
		0  5304.33      100.0            5.0

		"""

		ip, im = str(self.A)+self.element.upper(), self.isomer

		df = pd.read_sql('SELECT energy, intensity, unc_intensity FROM alphas WHERE isotope="{0}" AND isomer="{1}"'.format(ip, im), _get_connection('decay'))

		if I_lim is not None:
			if type(I_lim)==float or type(I_lim)==int:
				df = df[df['intensity']>=I_lim]
			else:
				df = df[(df['intensity']>=I_lim[0])&(df['intensity']<=I_lim[1])]

		if E_lim is not None:
			if type(E_lim)==float or type(E_lim)==int:
				df = df[df['energy']>=E_lim]
			else:
				df = df[(df['energy']>=E_lim[0])&(df['energy']<=E_lim[1])]

		return df.sort_values(by=['energy']).reset_index(drop=True)

	def dose_rate(self, activity=1.0, distance=30.0, units='R/hr'):
		"""Dose-rate from a point source of the isotope

		Assuming a point source with no attenuation, this function calculates the dose-rate
		from each emitted particle type using a specified source activity and distance.

		Parameters
		----------
		activity : float, optional
			Source activity in Bq. Default is 1.0

		distance : float, optional
			Source distance in cm. Default is 30.0

		units : str, optional
			Desired units of dose rate. Format: `dose/time` where dose is one of R, Sv, Gy, with
			the Si prefixes u, m, k, and M supported, and time is in the same format described under
			the `Isotope.half_life()` function. E.g. mR/hr, uSv/m, kGy/y. Default is 'R/hr'

		Returns
		-------
		dose_rate : dict
			Dictionary of dose from each of the following particle type: 'gamma', 'electron', 
			'beta_minus', 'beta_plus', 'alpha', and 'total'.  Units are specified as an input.

		Examples
		--------
		>>> ip = ci.Isotope('Co-60')
		>>> print(ip.dose_rate(activity=3.7E10, units='R/hr')) # 1 Ci of Co-60 at 30 cm
		{'beta_plus': 0.0, 'alphas': 0.0, 'gammas': 52035.28424827692, 
		'electrons': 65.66251805557148, 'beta_minus': 5536.541902410433, 'total': 57637.488668742924}

		"""

		def beta2(E_MeV, m_amu):
			return 1.0-(1.0/(1.0+(E_MeV/m_amu))**2)

		def e_range(E_keV):
			dEdx = lambda b2, t: 0.17*((np.log(3.61E5*t*np.sqrt(t+2)))+(0.5*(1-b2)*(1+t**2/8-(2*t+1)*np.log(2)))-4.312)/b2
			E = np.linspace(1E-12, E_keV*1E-3, 100)
			return np.trapz(1.0/((1.0+(E*7.22/800.0))*dEdx(beta2(E, 0.5109), E/0.5109)), E)

		def pos_range(E_keV):
			dEdx = lambda b2, t: 0.17*((np.log(3.61E5*t*np.sqrt(t+2)))+(np.log(2)-(b2/24)*(23+14/(t+2)+10/(t+2)**2+4/(t+2)**3))-4.312)/b2
			E = np.linspace(1E-12, E_keV*1E-3, 100)
			return np.trapz(1.0/((1.0+(E*7.22/800.0))*dEdx(beta2(E, 0.5109), E/0.5109)), E)

		def alpha_range(E_keV):
			dEdx = lambda b2: 0.17*4.0*((np.log(1.02E6*b2/(1.0-b2))-b2)-4.312)/b2
			return np.trapz(1.0/dEdx(beta2(np.linspace(1E-9, E_keV*1E-3, 100), 3.7284E3)), np.linspace(1E-9, E_keV*1E-3, 100))

		dose = {}

		gm = self.gammas(xrays=True, dE_511=0.0)
		al = self.alphas()
		bm = self.beta_minus()
		bp = self.beta_plus()
		el = self.electrons()

		dose['gammas'] = 1.4042E-12*np.sum(gm['energy']*gm['intensity'])*activity/distance**2

		dose['alphas'] = 5.2087E-14*np.sum([al['intensity'][n]*e/alpha_range(e) for n,e in enumerate(al['energy'].to_numpy())])*activity/distance**2

		dose['beta_minus'] = 5.2087E-14*np.sum([bm['intensity'][n]*e/e_range(e) for n,e in enumerate(bm['mean_energy'].to_numpy())])*activity/distance**2
		dose['beta_plus'] = 5.2087E-14*np.sum([bp['intensity'][n]*e/pos_range(e) for n,e in enumerate(bp['mean_energy'].to_numpy())])*activity/distance**2
		dose['electrons'] = 5.2087E-14*np.sum([el['intensity'][n]*e/e_range(e) for n,e in enumerate(el['energy'].to_numpy())])*activity/distance**2
		

		d_unit,t_unit = tuple(units.split('/'))

		if 'Sv' in d_unit:
			dose['alpha'] = 20.0*dose['alpha']
			d_unit = d_unit.replace('Sv','Gy')

		if t_unit in ['hr','min','yr','sec']:
			t_unit = {'hr':'h','min':'m','yr':'y','sec':'s'}[t_unit]

		t_conv = {'ns':1E-9, 'us':1E-6, 'ms':1E-3,
					's':1.0, 'm':60.0, 'h':3600.0,
					'd':86400.0, 'y':31557.6E3, 'ky':31557.6E6,
					'My':31557.6E9, 'Gy':31557.6E12}[t_unit]

		d_conv = {'R':1.0,'mR':1E3,'kR':1E-3,'uR':1E6,'MR':1E-6,
					'Gy':9.5E-3,'mGy':9.5,'kGy':9.5E-6,'uGy':9.5E3,'MGy':9.5E-9}[d_unit]

		dose = {i:d_conv*t_conv*dose[i] for i in ['gammas','alphas','beta_minus','beta_plus','electrons']}
		dose['total'] = sum([dose[i] for i in ['gammas','alphas','beta_minus','beta_plus','electrons']])

		return dose
