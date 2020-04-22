from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import json

from scipy.optimize import curve_fit

from .isotope import Isotope
from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap


class Calibration(object):
	"""Calibration for HPGe spectra

	Provides methods for calculating and fitting energy, efficiency and resolution
	calibrations for HPGe spectra.  Each Spectrum class contains a Calibration, 
	which can be loaded from a file, or fit to calibration spectra from known sources.
	
	Parameters
	----------
	filename : str, optional
		Path to a .json file storing calibration data.  The .json file can be produced by
		calling `cb.saveas('example_calib.json')` after performing a calibration fit with
		the `cb.calibrate()` function.  This allows past calibrations to be recalled without
		needing to re-perform the calibration fits.

	Attributes
	----------
	engcal : np.ndarray
		Energy calibration parameters. length 2 or 3 array, depending on whether the calibration
		is linear or quadratic.

	effcal : np.ndarray
		Efficiency calibration parameters. length 3 or 5 array, depending on whether the efficiency
		fit includes a "dead-layer term".

	unc_effcal : np.ndarray
		Efficiency calibration covariance matrix. shape 3x3 or 5x5, depending on the length of effcal.

	rescal : np.ndarray
		Resolution calibration parameters.  length 2 array if resolution calibration is of the form
		R = a + b*chan (default), or length 1 if R = a*sqrt(chan).


	Examples
	--------
	>>> cb = Calibration()
	>>> cb.engcal = [0.0, 0.25, 0.001]
	>>> print(cb.engcal)
	[0.0 0.3]
	>>> print(cb.effcal)
	[0.331 0.158 0.41 0.001 1.476]
	>>> cb.saveas('test_calib.json')
	>>> cb = Calibration('test_calib.json')
	>>> print(cb.engcal)
	[0.0 0.25 0.001]

	"""

	def __init__(self, filename=None):
		self._engcal = np.array([0.0, 0.3])
		self._effcal = np.array([0.331, 0.158, 0.410, 0.001, 1.476])
		self._unc_effcal = np.array([[ 5.038e-02,  3.266e-02, -2.151e-02, -4.869e-05, -7.748e-03],
									 [ 3.266e-02,  2.144e-02, -1.416e-02, -3.416e-05, -4.137e-03],
									 [-2.151e-02, -1.416e-02,  9.367e-03,  2.294e-05,  2.569e-03],
									 [-4.869e-05, -3.416e-05,  2.294e-05,  5.411e-07, -1.165e-04],
									 [-7.748e-03, -4.137e-03,  2.569e-03, -1.165e-04,  3.332e-02]])
		self._rescal = np.array([2.0, 4E-4])
		self._calib_data = {}

		if filename is not None:
			if filename.endswith('.json'):
				with open(filename) as f:
					js = json.load(f)
					if 'engcal' in js:
						self.engcal = js['engcal']
					if 'rescal' in js:
						self.rescal = js['rescal']
					if 'effcal' in js:
						self.effcal = js['effcal']
					if 'unc_effcal' in js:
						self.unc_effcal = js['unc_effcal']
					if '_calib_data' in js:
						for c in ['engcal','rescal','effcal']:
							if c in js['_calib_data']:
								self._calib_data[c] = {str(i):np.array(js['_calib_data'][c][i]) for i in js['_calib_data'][c]}


	@property
	def engcal(self):
		return self._engcal

	@engcal.setter
	def engcal(self, cal):
		self._engcal = np.asarray(cal)

	@property
	def effcal(self):
		return self._effcal

	@effcal.setter
	def effcal(self, cal):
		self._effcal = np.asarray(cal)

	@property
	def unc_effcal(self):
		return self._unc_effcal

	@unc_effcal.setter
	def unc_effcal(self, cal):
		self._unc_effcal = np.asarray(cal)

	@property
	def rescal(self):
		return self._rescal

	@rescal.setter
	def rescal(self, cal):
		self._rescal = np.asarray(cal)


	def eng(self, channel, engcal=None):
		"""Energy calibration function

		Returns the calculated energy given an input array of
		channel numbers.  The `engcal` can be supplied, or
		if `engcal=None` the Calibration object's energy
		calibration is used (cb.engcal).

		Parameters
		----------
		channel : array_like
			Spectrum channel number.  The maximum channel number should
			be the length of the spectrum.

		engcal : array_like, optional
			Optional energy calibration. If a length 2 array, calibration
			will be engcal[0] + engcal[1]*channel.  If length 3, then
			engcal[0] + engcal[1]*channel + engcal[2]*channel**2

		Returns
		-------
		energy: np.ndarray
			Calibrated energy corresponding to the given channels.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.engcal)
		[0.  0.3]
		>>> print(cb.eng(np.arange(10)))
		[0.  0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7]
		>>> cb.engcal = [0.1, 0.2, 0.003]
		>>> print(cb.eng(np.arange(10)))
		[0.1   0.303 0.512 0.727 0.948 1.175 1.408 1.647 1.892 2.143]

		"""

		channel = np.asarray(channel)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==2:
			return engcal[0] + engcal[1]*channel

		elif len(engcal)==3:
			return engcal[0] + engcal[1]*channel + engcal[2]*channel**2
		
	def eff(self, energy, effcal=None):
		"""Efficiency calibration function

		Returns the calculated (absolute) efficiency given an input array of 
		energies. The `effcal` can be supplied, or if `effcal=None`, the
		calibration object's internal efficiency calibration (cb.effcal)
		is used.

		The functional form of the efficiency used is
		eff(E) = c[0]*exp(-c[1]*energy**c[2]))) if the effcal is length 3 or
		eff(E) = c[0]*exp(-c[1]*energy**c[2])))*(1-exp(-c[3]*energy**c[4])) if the
		effcal is length 5.

		Parameters
		----------
		energy : array_like
			Peak energy in keV.

		effcal : array_like, optional
			Efficiency calibration parameters. length 3 or 5 array, depending on whether the efficiency
			fit includes a "dead-layer term".

		Returns
		-------
		efficiency : np.ndarray
			Absolute efficiency at the given energies.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.effcal)
		[3.310e-01 1.580e-01 4.100e-01 1.000e-03 1.476e+00]
		>>> print(cb.eff(50*np.arange(1,10)))
		[0.04152215 0.06893742 0.07756411 0.07583971 0.07013108 0.06365222
		 0.05762826 0.05236502 0.0478359 ]

		"""

		energy = np.asarray(energy)

		if effcal is None:
			effcal = self.effcal

		if len(effcal)==3:
			return effcal[0]*np.exp(-effcal[1]*energy**effcal[2])

		elif len(effcal)==5:
			return effcal[0]*np.exp(-effcal[1]*energy**effcal[2])*(1.0-np.exp(-effcal[3]*energy**effcal[4]))
		
	def unc_eff(self, energy, effcal=None, unc_effcal=None):
		"""Uncertainty in the efficiency

		Returns the calculated uncertainty in efficiency for an input
		array of energies.  If `effcal` or `unc_effcal` are not none,
		they are used instead of the calibration object's internal
		values. (cb.unc_effcal)  unc_effcal must be a covariance matrix of the
		same dimension as effcal.

		Parameters
		----------
		energy : array_like
			Peak energy in keV.

		effcal : array_like, optional
			Efficiency calibration parameters. length 3 or 5 array, depending on whether the efficiency
			fit includes a "dead-layer term".

		unc_effcal : array_like, optional
			Efficiency calibration covariance matrix. shape 3x3 or 5x5, depending on the length of effcal.

		Returns
		-------
		unc_efficiency : np.ndarray
			Absolute uncertainty in efficiency for the given energies.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.effcal)
		[3.310e-01 1.580e-01 4.100e-01 1.000e-03 1.476e+00]
		>>> print(cb.unc_effcal)
		[[ 5.038e-02  3.266e-02 -2.151e-02 -4.869e-05 -7.748e-03]
		 [ 3.266e-02  2.144e-02 -1.416e-02 -3.416e-05 -4.137e-03]
		 [-2.151e-02 -1.416e-02  9.367e-03  2.294e-05  2.569e-03]
		 [-4.869e-05 -3.416e-05  2.294e-05  5.411e-07 -1.165e-04]
		 [-7.748e-03 -4.137e-03  2.569e-03 -1.165e-04  3.332e-02]]
		>>> print(cb.eff(50*np.arange(1,10)))
		[0.04152215 0.06893742 0.07756411 0.07583971 0.07013108 0.06365222
		 0.05762826 0.05236502 0.0478359 ]
		>>> print(cb.unc_eff(50*np.arange(1,10)))
		[0.00453714 0.00795861 0.00705099 0.00514765 0.00453579 0.00433504
		 0.00394935 0.00347228 0.00304041]

		"""

		energy = np.asarray(energy)

		if effcal is None or unc_effcal is None:
			effcal, unc_effcal = self.effcal, self.unc_effcal

		eps = 1E-8
		var = np.zeros(len(energy)) if energy.shape else 0.0

		for n in range(len(effcal)):
			for m in range(n, len(effcal)):

				if not np.isfinite(unc_effcal[n][m]):
					return np.inf*(var+1.0)

				c_n, c_m = np.copy(effcal), np.copy(effcal)
				c_n[n], c_m[m] = c_n[n]+eps, c_m[m]+eps

				par_n = (self.eff(energy, c_n)-self.eff(energy, effcal))/eps
				par_m = (self.eff(energy, c_m)-self.eff(energy, effcal))/eps

				var += unc_effcal[n][m]*par_n*par_m*(2.0 if n!=m else 1.0)

		return np.sqrt(var)
		
	def res(self, channel, rescal=None):
		"""Resolution calibration

		Calculates the expected 1-sigma peak widths for a given input array
		of channel numbers.  If `rescal` is given, it is used instead of
		the calibration object's internal value (cb.rescal).

		Parameters
		----------
		channel : array_like
			Spectrum channel number.  The maximum channel number should
			be the length of the spectrum.

		rescal : array_like, optional
			Resolution calibration parameters.  length 2 array if resolution calibration is of the form
			R = a + b*chan (default), or length 1 if R = a*sqrt(chan).

		Returns
		-------
		resolution : np.ndarray
			Calculated 1-sigma width of the peaks given the input channel numbers.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.rescal)
		[2.e+00 4.e-04]
		>>> print(cb.res(100*np.arange(1,10)))
		[2.04 2.08 2.12 2.16 2.2  2.24 2.28 2.32 2.36]

		"""

		channel = np.asarray(channel)

		if rescal is None:
			rescal = self.rescal

		if len(rescal)==1:
			return rescal[0]*np.sqrt(channel)

		elif len(rescal)==2:
			return rescal[0] + rescal[1]*channel
		
	def map_channel(self, energy, engcal=None):
		"""Energy to channel calibration

		Calculates the spectrum channel number corresponding to a given
		energy array.  This should return the inverse of the energy calibration, but
		as an integer-type channel number.

		Parameters
		----------
		energy : array_like
			Peak energy in keV

		engcal : array_like, optional
			Energy calibration parameters. length 2 or 3 array, depending on whether the calibration
			is linear or quadratic.

		Returns
		-------
		channel : np.ndarray
			Calculated channel number given the input energy array

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.engcal)
		[0.  0.3]
		>>> print(cb.map_channel(300))
		1000
		>>> print(cb.eng(cb.map_channel(300)))
		300.0

		"""

		energy = np.asarray(energy)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==3:
			if engcal[2]!=0.0:
				return np.array(np.rint(0.5*(np.sqrt(engcal[1]**2-4.0*engcal[2]*(engcal[0]-energy))-engcal[1])/engcal[2]), dtype=np.int32)

		return np.array(np.rint((energy-engcal[0])/engcal[1]), dtype=np.int32)
		
	def calibrate(self, spectra, sources):
		"""Generate calibration parameters from spectra

		Performs an energy, resolution and efficiency calibration on peak fits
		to a given list of spectra.  Reference activities must be given for the
		efficiency calibration.  Spectra are allowed to have isotopes that are
		not in `sources`, but these will not be included in the efficiency calibration.

		Parameters
		----------
		spectra : list of sp.Spectrum
			List of calibration spectra.  Must have sp.isotopes defined, and matching
			the isotopes given in `sources`.

		sources : str, list, dict or pd.DataFrame
			Datatype or (if str) file that can be converted into a
			pandas DataFrame.  Required keys are 'isotope', 'A0' (reference activity),
			and 'ref_date' (reference date).

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> print(cb.effcal)
		[4.33742771 2.28579733 0.15337749]
		>>> cb.plot()

		"""

		if type(sources)==str:
			if sources.endswith('.json'):
				sources = pd.DataFrame(json.loads(open(sources).read()))
			elif sources.endswith('.csv'):
				sources = pd.read_csv(sources, header=0).fillna(method='ffill')
			elif sources.endswith('.db'):
				sources = pd.read_sql('SELECT * FROM sources', _get_connection(sources))
		else:
			sources = pd.DataFrame(sources)
		sources['ref_date'] = pd.to_datetime(sources['ref_date'], format='%m/%d/%Y %H:%M:%S')

		self._calib_data = {'engcal':{'channel':[], 'energy':[], 'unc_channel':[]},
							'rescal':{'channel':[], 'width':[], 'unc_width':[]},
							'effcal':{'energy':[], 'efficiency':[], 'unc_efficiency':[]}}

		src_itps = sources['isotope'].to_list()
		lm = {ip:Isotope(ip).dc for ip in src_itps}
		unc_lm = {ip:Isotope(ip).decay_const(unc=True)[1] for ip in src_itps}

		specs = []
		for n,sp in enumerate(spectra):
			if type(sp)==str:
				from .spectrum import Spectrum
				spec = Spectrum(sp)
				spec.isotopes = sources['isotope'].to_list()
				specs.append(spec)
			else:
				specs.append(sp)
		spectra = specs

		for sp in spectra:
			if sp._fits is None:
				sp.fit_peaks()

			cfg = sp.fit_config
			ix = -1

			for ft in sp._fits:
				f = ft['fit']
				u = ft['unc']
				B = {'snip':0, 'constant':1, 'linear':2, 'quadratic':3}[cfg['bg'].lower()]
				L = 3+2*int(cfg['skew_fit'])+int(cfg['step_fit'])

				for n in range(int((len(f)-B)/L)):
					ix += 1
					pk = sp.peaks.loc[ix, :]
					if pk['isotope'] not in src_itps:
						continue

					j = B+n*L+1
					mu, sig = f[j], f[j+1]
					unc_mu, unc_sig = np.sqrt(u[j][j]), np.sqrt(u[j+1][j+1])
					eng = pk['energy']

					self._calib_data['engcal']['channel'].append(mu)
					self._calib_data['engcal']['energy'].append(eng)
					self._calib_data['engcal']['unc_channel'].append(unc_mu)

					self._calib_data['rescal']['channel'].append(mu)
					self._calib_data['rescal']['width'].append(sig)
					self._calib_data['rescal']['unc_width'].append(unc_sig)

					src = sources[sources['isotope']==pk['isotope']].reset_index(drop=True)
					rd, A0 = src.loc[0, 'ref_date'], src.loc[0, 'A0']
					unc_A0 = src.loc[0, 'unc_A0'] if 'unc_A0' in src.columns else 0.0
					td = (sp.start_time-rd).total_seconds()
					dc = lm[pk['isotope']]

					if sp._atten_corr is None:
						corr = sp._geom_corr
					else:
						corr = sp._geom_corr*sp._atten_corr(pk['energy'])

					eff = pk['counts']*dc/(corr*(1.0-np.exp(-dc*sp.real_time))*np.exp(-dc*td)*pk['intensity']*A0*(sp.live_time/sp.real_time))
					unc_eff = eff*np.sqrt((pk['unc_counts']/pk['counts'])**2+(pk['unc_intensity']/pk['intensity'])**2+(unc_lm[pk['isotope']]/dc)**2+(unc_A0/A0)**2)

					self._calib_data['effcal']['energy'].append(eng)
					self._calib_data['effcal']['efficiency'].append(eff)
					self._calib_data['effcal']['unc_efficiency'].append(unc_eff)

		for ctyp in self._calib_data:
			for cvar in self._calib_data[ctyp]:
				self._calib_data[ctyp][cvar] = np.array(self._calib_data[ctyp][cvar])

		x, y, yerr = self._calib_data['engcal']['channel'], self._calib_data['engcal']['energy'], self.eng(self._calib_data['engcal']['unc_channel'])
		idx = np.where((0.25*y>yerr)&(yerr>0.0)&(np.isfinite(yerr)))
		x, y, yerr = x[idx], y[idx], yerr[idx]
		fn = lambda x, *A: self.eng(x, A)
		fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=spectra[0].cb.engcal)
		self._calib_data['engcal']['channel'], self._calib_data['engcal']['energy'], self._calib_data['engcal']['unc_channel'] = x, y, yerr
		self._calib_data['engcal']['fit'], self._calib_data['engcal']['unc'] = fit, unc

		x, y, yerr = self._calib_data['rescal']['channel'], self._calib_data['rescal']['width'], self._calib_data['rescal']['unc_width']
		idx = np.where((0.25*y>yerr)&(yerr>0.0)&(np.isfinite(yerr)))
		x, y, yerr = x[idx], y[idx], yerr[idx]
		fn = lambda x, *A: self.res(x, A)
		fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=spectra[0].cb.rescal)
		self._calib_data['rescal']['channel'], self._calib_data['rescal']['width'], self._calib_data['rescal']['unc_width'] = x, y, yerr
		self._calib_data['rescal']['fit'], self._calib_data['rescal']['unc'] = fit, unc

		x, y, yerr = self._calib_data['effcal']['energy'], self._calib_data['effcal']['efficiency'], self._calib_data['effcal']['unc_efficiency']
		idx = np.where((0.25*y>yerr)&(yerr>0.0)&(np.isfinite(yerr)))
		x, y, yerr = x[idx], y[idx], yerr[idx]
		fn = lambda x, *A: self.eff(x, A)

		p0 = spectra[0].cb.effcal
		p0 = p0.tolist() if len(p0)==5 else p0.tolist()+[0.001, 1.476]
		p0[0] = max([min([p0[0]*np.average(y/self.eff(x, p0), weights=(self.eff(x, p0)/yerr)**2),99.9]),0.001])
		bounds = ([0.0, 0.0, -1.0, 0.0, -2.0], [100.0, 3.0, 3.0, 0.5, 3.0])

		if any([sp.fit_config['xrays'] for sp in spectra]):
			try:
				fit3, unc3 = curve_fit(fn, x, y, sigma=yerr, p0=p0[:3], bounds=(bounds[0][:3], bounds[1][:3]))
				fit5, unc5 = curve_fit(fn, x, y, sigma=yerr, p0=fit3.tolist()+p0[3:], bounds=bounds)
				
				chi5 = np.sum((y-self.eff(x, fit5))**2/yerr**2)
				chi3 = np.sum((y-self.eff(x, fit3))**2/yerr**2)
				## Invert to find which is closer to one
				chi5 = chi5 if chi5>1.0 else 1.0/chi5
				chi3 = chi3 if chi3>1.0 else 1.0/chi3
				fit, unc = (fit3, unc3) if chi3<=chi5 else (fit5, unc5)
			except:
				fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=p0[:3], bounds=(bounds[0][:3], bounds[1][:3]))

		else:
			fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=p0[:3], bounds=(bounds[0][:3], bounds[1][:3]))

		idx = np.where((fn(x, *fit)-y)**2/yerr**2 < 10.0)
		x, y, yerr = x[idx], y[idx], yerr[idx]
		self._calib_data['effcal']['energy'], self._calib_data['effcal']['efficiency'], self._calib_data['effcal']['unc_efficiency'] = x, y, yerr

		self._calib_data['effcal']['fit'], self._calib_data['effcal']['unc'] = fit, unc

		self.engcal = self._calib_data['engcal']['fit']
		self.rescal = self._calib_data['rescal']['fit']
		self.effcal = self._calib_data['effcal']['fit']
		self.unc_effcal = self._calib_data['effcal']['unc']

		for sp in spectra:
			sp.cb.engcal = self._calib_data['engcal']['fit']
			sp.cb.rescal = self._calib_data['rescal']['fit']
			sp.cb.effcal = self._calib_data['effcal']['fit']
			sp.cb.unc_effcal = self._calib_data['effcal']['unc']
			sp._peaks, sp._fits = None, None



		
	def saveas(self, filename):
		"""Save the calibration as a .json file

		Saves the energy, resolution and efficiency calibration to a .json
		file, which can be recalled by a new calibration object by passing
		the 'filename' keyword to Calibration() upon construction.

		Parameters
		----------
		filename : str
			Complete filename to save the calibration. Must end in '.json'.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> print(cb.effcal)
		[4.33742771 2.28579733 0.15337749]
		>>> cb.saveas('example_calib.json')

		>>> cb = ci.Calibration('example_calib.json')
		>>> print(cb.effcal)
		[4.33742771 2.28579733 0.15337749]

		"""

		if any([filename.endswith(e) for e in ['.png','.pdf','.eps','.pgf','.ps','.raw','.rgba','.svg','.svgz']]):
				self.plot(saveas=filename, show=False)

		if filename.endswith('.json'):
			js = {'engcal':self.engcal.tolist(),
				  'effcal':self.effcal.tolist(),
				  'unc_effcal':self.unc_effcal.tolist(),
				  'rescal':self.rescal.tolist()}
			if self._calib_data:
				js['_calib_data'] = {}
				for cl in ['engcal','rescal','effcal']:
					if cl in self._calib_data:
						js['_calib_data'][cl] = {i:self._calib_data[cl][i].tolist() for i in self._calib_data[cl]}

			with open(filename, 'w') as f:
				json.dump(js, f, indent=4)

		
	def plot_engcal(self, **kwargs):
		"""Plot the energy calibration

		Draws the energy calibration, with measurements from peak fit data if available.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_engcal()

		"""

		f, ax = _init_plot(**kwargs)

		if 'engcal' in self._calib_data:
			d = self._calib_data['engcal']
			ax.errorbar(d['energy'], d['channel'], yerr=d['unc_channel'], ls='None', marker='o')
			x = np.arange(min(d['energy']), max(d['energy']), 0.1)
			ax.plot(x, self.map_channel(x, d['fit']))

		else:
			x = np.arange(20, 3000, 0.1)
			ax.plot(x, self.map_channel(x))

		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('ADC Channel')
		ax.set_title('Energy Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_rescal(self, **kwargs):
		"""Plot the resolution calibration

		Draws the resolution calibration, with measurements from peak fit data if available.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_rescal()

		"""

		f, ax = _init_plot(**kwargs)

		if 'rescal' in self._calib_data:
			d = self._calib_data['rescal']
			ax.errorbar(d['channel'], d['width'], yerr=d['unc_width'], ls='None', marker='o')
			x = np.arange(min(d['channel']), max(d['channel']), 0.1)
			ax.plot(x, self.res(x, d['fit']))

		else:
			x = np.arange(0, 2**14, 0.1)
			ax.plot(x, self.res(x))

		ax.set_xlabel('ADC Channel')
		ax.set_ylabel('Peak Width')
		ax.set_title('Resolution Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_effcal(self, **kwargs):
		"""Plot the efficiency calibration

		Draws the efficiency calibration, with measurements from peak fit data if available.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_effcal()

		"""
		cm = colormap()
		cm_light = colormap(shade='light')

		f, ax = _init_plot(**kwargs)

		if 'effcal' in self._calib_data:
			d = self._calib_data['effcal']
			ax.errorbar(d['energy'], d['efficiency'], yerr=d['unc_efficiency'], ls='None', marker='o', color=cm['k'])
			x = np.arange(min(d['energy']), max(d['energy']), 0.1)
			ax.plot(x, self.eff(x, d['fit']), color=cm['k'])
			low = self.eff(x, d['fit'])-self.unc_eff(x, d['fit'], d['unc'])
			high = self.eff(x, d['fit'])+self.unc_eff(x, d['fit'], d['unc'])
			ax.fill_between(x, low, high, facecolor=cm_light['k'], alpha=0.5)

		else:
			x = np.arange(20, 3000, 0.1)
			ax.plot(x, self.eff(x), color=cm['k'])
			low = self.eff(x)-self.unc_eff(x)
			high = self.eff(x)+self.unc_eff(x)
			ax.fill_between(x, low, high, facecolor=cm_light['k'], alpha=0.5)


		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('Efficiency')
		ax.set_title('Efficiency Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot(self, **kwargs):
		"""Plots energy, resolution and efficiency calibrations

		Draws all three of energy, resolution and efficiency calibrations on a 
		single figure, and shows measured values from peak data, if available.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot()

		"""

		
		f, ax = _init_plot(_N_plots=3, figsize=(12.8, 4.8), **kwargs)

		f, ax[0] = self.plot_engcal(f=f, ax=ax[0], show=False, return_plot=True)
		f, ax[1] = self.plot_rescal(f=f, ax=ax[1], show=False, return_plot=True)
		f, ax[2] = self.plot_effcal(f=f, ax=ax[2], show=False, return_plot=True)

		return _draw_plot(f, ax, **kwargs)
		

