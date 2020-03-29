from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import json

from scipy.optimize import curve_fit

from .isotope import Isotope
from .plotting import _init_plot, _draw_plot, colormap


class Calibration(object):
	"""Calibration

	...
	
	Parameters
	----------
	filename : str, optional
		Description of parameter `x`.

	Attributes
	----------
	engcal : np.ndarray
		Description of parameter

	effcal : np.ndarray
		Description of parameter

	unc_effcal : np.ndarray
		Description of parameter

	rescal : np.ndarray
		Description of parameter


	Examples
	--------

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
		""" Description

		...

		Parameters
		----------
		channel : array_like
			Description of x

		engcal : array_like, optional
			Description

		Returns
		-------
		type
			Description

		Examples
		--------

		"""

		channel = np.asarray(channel)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==2:
			return engcal[0] + engcal[1]*channel

		elif len(engcal)==3:
			return engcal[0] + engcal[1]*channel + engcal[2]*channel**2
		
	def eff(self, energy, effcal=None):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		effcal : array_like, optional
			Description of x

		Returns
		-------
		type
			Description

		Examples
		--------

		"""

		energy = np.asarray(energy)

		if effcal is None:
			effcal = self.effcal

		if len(effcal)==3:
			return effcal[0]*np.exp(-effcal[1]*energy**effcal[2])

		elif len(effcal)==5:
			return effcal[0]*np.exp(-effcal[1]*energy**effcal[2])*(1.0-np.exp(-effcal[3]*energy**effcal[4]))
		
	def unc_eff(self, energy, effcal=None, unc_effcal=None):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		effcal : array_like, optional
			Description of x

		unc_effcal : array_like, optional
			Description of x

		Returns
		-------
		type
			Description

		Examples
		--------

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
		""" Description

		...

		Parameters
		----------
		channel : array_like
			Description of x

		rescal : array_like, optional
			Description of x

		Returns
		-------
		type
			Description

		Examples
		--------

		"""

		channel = np.asarray(channel)

		if rescal is None:
			rescal = self.rescal

		if len(rescal)==1:
			return rescal[0]*np.sqrt(channel)

		elif len(rescal)==2:
			return rescal[0] + rescal[1]*channel
		
	def map_channel(self, energy, engcal=None):
		""" Description

		...

		Parameters
		----------
		energy : array_like
			Description of x

		engcal : array_like, optional
			Description of x

		Returns
		-------
		type
			Description

		Examples
		--------

		"""

		energy = np.asarray(energy)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==3:
			if engcal[2]!=0.0:
				return np.array(np.rint(0.5*(np.sqrt(engcal[1]**2-4.0*engcal[2]*(engcal[0]-energy))-engcal[1])/engcal[2]), dtype=np.int32)

		return np.array(np.rint((energy-engcal[0])/engcal[1]), dtype=np.int32)
		
	def calibrate(self, spectra, sources):
		""" Description

		...

		Parameters
		----------
		spectra : list of sp.Spectrum
			Description of x

		sources : str, list, dict or pd.DataFrame
			Description

		Examples
		--------

		"""

		if type(sources)==str:
			if sources.endswith('.json'):
				sources = pd.DataFrame(json.loads(open(sources).read()))
			elif sources.endswith('.csv'):
				sources = pd.read_csv(sources, header=0).fillna(method='ffill')
		else:
			sources = pd.DataFrame(sources)
		sources['ref_date'] = pd.to_datetime(sources['ref_date'], format='%m/%d/%Y %H:%M:%S')

		self._calib_data = {'engcal':{'channel':[], 'energy':[], 'unc_channel':[]},
							'rescal':{'channel':[], 'width':[], 'unc_width':[]},
							'effcal':{'energy':[], 'efficiency':[], 'unc_efficiency':[]}}

		src_itps = sources['isotope'].to_list()
		lm = {ip:Isotope(ip).dc for ip in src_itps}
		unc_lm = {ip:Isotope(ip).decay_const(unc=True)[1] for ip in src_itps}
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

					src = sources[sources['isotope']==pk['isotope']]
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
		fn = lambda x, *A: self.eng(x, A)
		fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=spectra[0].cb.engcal)
		self._calib_data['engcal']['fit'], self._calib_data['engcal']['unc'] = fit, unc

		x, y, yerr = self._calib_data['rescal']['channel'], self._calib_data['rescal']['width'], self._calib_data['rescal']['unc_width']
		fn = lambda x, *A: self.res(x, A)
		fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=spectra[0].cb.rescal)
		self._calib_data['rescal']['fit'], self._calib_data['rescal']['unc'] = fit, unc

		x, y, yerr = self._calib_data['effcal']['energy'], self._calib_data['effcal']['efficiency'], self._calib_data['effcal']['unc_efficiency']
		idx = np.where((y>yerr)&(yerr>0.0)&(np.isfinite(yerr)))
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
		""" Description

		...

		Parameters
		----------
		filename : str
			Description of x

		Examples
		--------

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
		""" Description

		...

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

		if 'engcal' in self._calib_data:
			d = self._calib_data['engcal']
			ax.errorbar(d['energy'], d['channel'], yerr=d['unc_channel'], ls='None', marker='o')
			x = np.arange(min(d['energy']), max(d['energy']), 0.1)
			ax.plot(x, self.map_channel(x, d['fit']))

		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('ADC Channel')
		ax.set_title('Energy Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_rescal(self, **kwargs):
		""" Description

		...

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

		if 'rescal' in self._calib_data:
			d = self._calib_data['rescal']
			ax.errorbar(d['channel'], d['width'], yerr=d['unc_width'], ls='None', marker='o')
			x = np.arange(min(d['channel']), max(d['channel']), 0.1)
			ax.plot(x, self.res(x, d['fit']))

		ax.set_xlabel('ADC Channel')
		ax.set_ylabel('Peak Width')
		ax.set_title('Resolution Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_effcal(self, **kwargs):
		""" Description

		...

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------

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

		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('Efficiency')
		ax.set_title('Efficiency Calibration')

		return _draw_plot(f, ax, **kwargs)
		
	def plot(self, **kwargs):
		""" Description

		...

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------

		"""

		
		f, ax = _init_plot(_N_plots=3, figsize=(12.8, 4.8), **kwargs)

		f, ax[0] = self.plot_engcal(f=f, ax=ax[0], show=False, return_plot=True)
		f, ax[1] = self.plot_rescal(f=f, ax=ax[1], show=False, return_plot=True)
		f, ax[2] = self.plot_effcal(f=f, ax=ax[2], show=False, return_plot=True)

		return _draw_plot(f, ax, **kwargs)
		

