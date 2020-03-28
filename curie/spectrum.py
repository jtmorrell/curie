from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import pandas as pd
import datetime as dtm
import copy

from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.interpolate import interp1d

from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from .calibration import Calibration
from .isotope import Isotope
from .compound import Compound

class Spectrum(object):
	"""Spectrum

	...
	
	Parameters
	----------
	filename : str
		Description of parameter `x`.

	Other Parameters
	----------------
	cb : str or ci.Calibration
		Description

	isotopes : list
		Description

	fit_config : dict
		Description

	Attributes
	----------
	cb : ci.Calibration
		Description

	isotopes : list
		Description

	fit_config : dict
		Description

	filename : str
		Description

	hist : np.ndarray
		Description

	start_time : datetime.datetime
		Description

	live_time : float
		Description

	real_time : float
		Description

	peaks : pd.DataFrame
		Description


	Examples
	--------

	"""

	def __init__(self, filename, **kwargs):
		self.filename = filename
		if 'cb' in kwargs:
			if type(kwargs['cb']==str):
				self._cb = Calibration(kwargs['cb'])
			else:
				self._cb = copy.deepcopy(kwargs['cb'])
		else:
			self._cb = Calibration()

		self._fit_config = {'snip_adj':1.0, 'R':0.1, 'alpha':0.9,
							'step':0.00, 'bg':'snip', 'skew_fit':False,
							'step_fit':False, 'SNR_min':4.0, 'A_max':10.0,
							'mu_max':1.5, 'sig_max':1.5, 'xrays':False,
							'pk_width':7.5, 'E_min':75.0, 'I_min':0.05,
							'dE_511':3.5, 'multi_max':8}
		if 'fit_config' in kwargs:
			self.fit_config = kwargs['fit_config']

		if filename is not None:
			if os.path.exists(filename):
				print('Reading Spectrum {}'.format(filename))
				if filename.endswith('.Spe'):
					self._read_Spe(filename)
				elif filename.endswith('.Chn'):
					self._read_Chn(filename)
				else:
					raise ValueError('File type not supported: {}'.format(filename))
				self._snip = self._snip_bg()
				self._snip_interp = interp1d(np.arange(len(self.hist)), self._snip, bounds_error=False, fill_value=0.0)
			else:
				raise ValueError('File does not exist: {}'.format(filename))

		if 'isotopes' in kwargs:
			self.isotopes = kwargs['isotopes']
		else:
			self.isotopes = []
		self._peaks = None
		self._fits = None
		self._geom_corr = 1.0
		self._atten_corr = None

		self._gmls = None

	def _read_Spe(self, filename):
		self._ortec_metadata = {}

		with open(filename) as f:
			ln = f.readline()
			while ln:
				if ln.startswith('$'):
					section = ln.strip()[1:-1]
					if section=='DATA':
						L = int(f.readline().strip().split()[1])+1
						self.hist = np.fromfile(f, dtype=np.int64, count=L, sep='\n')
					else:
						self._ortec_metadata[section] = []
				else:
					self._ortec_metadata[section].append(ln.strip())
				ln = f.readline()

		self.start_time = dtm.datetime.strptime(self._ortec_metadata['DATE_MEA'][0], '%m/%d/%Y %H:%M:%S')
		self.live_time, self.real_time = tuple(map(float, self._ortec_metadata['MEAS_TIM'][0].split()))

		self.cb.engcal = list(map(float, self._ortec_metadata['MCA_CAL'][-1].split(' ')[:-1]))

	def _read_Chn(self, filename):
		self._ortec_metadata = {}

		with open(filename, 'rb') as f:
			det_no = np.frombuffer(f.read(6), dtype='i2')[1]
			sts = np.frombuffer(f.read(2), dtype='S2')[0].decode('utf-8')

			self.real_time, self.live_time = tuple(map(float, 0.02*np.frombuffer(f.read(8), dtype='i4')))

			st = np.frombuffer(f.read(12), dtype='S12')[0].decode('utf-8')
			months = {'jan':'01','feb':'02','mar':'03','apr':'04',
						'may':'05','jun':'06','jul':'07','aug':'08',
						'sep':'09','oct':'10','nov':'11','dec':'12'}
			start_time = '{0}/{1}/{2} {3}:{4}:{5}'.format(months[st[2:5].lower()],st[:2],('20' if st[7]=='1' else '19')+st[5:7],st[8:10],st[10:],sts)
			self.start_time = dtm.datetime.strptime(start_time, '%m/%d/%Y %H:%M:%S')

			L = np.frombuffer(f.read(4), dtype='i2')[1]
			self.hist = np.asarray(np.frombuffer(f.read(4*L), dtype='i4'), dtype=np.int64)
			f.read(4)

			self.cb.engcal = np.frombuffer(f.read(12),dtype='f4').tolist()


			shape = np.frombuffer(f.read(12), dtype='f4').tolist()
			self._ortec_metadata['SHAPE_CAL'] = ['3', ' '.join(map(str, shape))]
			f.read(228)
			L = np.frombuffer(f.read(1), dtype='i1')[0]
			if L:
				det_desc = np.frombuffer(f.read(L), dtype='S{}'.format(L))[0].decode('utf-8')
				self._ortec_metadata['SPEC_REM'] = ['DET# '+str(det_no), 'DETDESC# '+det_desc, 'AP# Maestro Version 7.01']
			if L<63:
				f.read(63-L)
			L = np.frombuffer(f.read(1), dtype='i1')[0]
			if L:
				sample_desc = np.frombuffer(f.read(L),dtype='S{}'.format(L))[0].decode('utf-8')
				self._ortec_metadata['SPEC_ID'] = [sample_desc]


	@property
	def cb(self):
		return self._cb

	@cb.setter
	def cb(self, _cb):
		if type(_cb)==str:
			self._cb = Calibration(_cb)
		else:
			self._cb = copy.deepcopy(_cb)


	@property
	def fit_config(self):
		return self._fit_config

	@fit_config.setter
	def fit_config(self, _fit_config):
		for nm in _fit_config:
			self._fit_config[nm] = _fit_config[nm]


	def __str__(self):
		return self.filename

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

		if not type(other)==Spectrum:
			raise ValueError('Cannot add spectrum to type {}'.format(type(other)))

		if self.start_time==other.start_time:
			alpha = np.sum(self.hist)/float(np.sum(other.hist))
			dead_time = alpha*(self.real_time-self.live_time)+(1.0-alpha)*(other.real_time-other.live_time)
			self.real_time = alpha*self.real_time+(1.0-alpha)*other.real_time
			self.live_time = self.real_time-dead_time
		else:
			self.real_time += other.real_time
			self.live_time += other.live_time
			

		if len(self.hist)==len(other.hist):
			if len(self.cb.engcal)==len(other.cb.engcal):
				if len(np.where(self.cb.engcal==other.cb.engcal)[0])==len(self.cb.engcal):
					self.hist += other.hist
					return self

		other_bins = other.cb.eng(np.arange(-0.5, len(other.hist)+0.5, 1.0))
		dNdE = np.array(other.hist, dtype=np.float64)/(other_bins[1:]-other_bins[:-1])
		f = interp1d(other.cb.eng(np.arange(len(other.hist))), dNdE, bounds_error=False, fill_value=0.0)

		bins = self.cb.eng(np.arange(-0.5, len(self.hist)+0.5, 1.0))
		edges = np.append(bins, bins).reshape((2,len(bins))).T.flatten()[1:-1].reshape((len(bins)-1, 2)).T
		if np.__version__>='1.16.0':
			e_grid = np.linspace(edges[0], edges[1], num=10).T
		else:
			e_grid = np.array([np.linspace(edges[0][n], edges[1][n], num=10) for n in range(len(edges[0]))])
		N = np.asarray(np.trapz(f(e_grid), e_grid, axis=1), dtype=np.int64)
		self.hist += np.random.poisson(N)

		return self
		
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

		L = len(self.hist)
		if N_bins>L:
			raise ValueError('N_bins: {0} must not be greater than current value: {1}'.format(N_bins, L))

		r = int(round(L/float(N_bins)))
		self.hist = np.sum(self.hist.reshape((int(L/r), r)), axis=1)

		ec = self.cb.engcal
		self.cb.engcal = [ec[0], ec[1]*r]+([ec[2]*r**2] if len(ec)==3 else [])

		rc = self.cb.rescal
		self.cb.rescal = ([rc[0]*np.sqrt(r)] if len(rc)==1 else [rc[0], rc[1]*r])
		
	def attenuation_correction(self, compounds, x=None, ad=None): # first compound is 'self'
		""" Description

		...

		Parameters
		----------
		compounds : list of str or ci.Compound
			Description of x

		x : list of float
			Description

		ad : list of float
			Description

		Examples
		--------

		"""

		cm_ls = [Compound(cm) if type(cm)==str else cm for cm in compounds]
		energy = np.unique(np.concatenate([cm.mass_coeff['energy'].to_numpy() for cm in cm_ls]))

		if ad is None:
			ad = [x[n]*cm.density for n,cm in enumerate(cm_ls)]
		ad = np.asarray(ad)

		mu_0 = cm_ls[0].mu(energy)
		# atten = (1.0-(ad[0]*mu_0+1.0)*np.exp(-ad[0]*mu_0))/ad[0]**2
		atten = (1.0-np.exp(-ad[0]*mu_0))/(ad[0]*mu_0)

		for n,cm in enumerate(cm_ls[1:]):
			atten *= cm.attenuation(energy, ad[n+1], 1.0)

		atten_corr = interp1d(energy, atten, bounds_error=False, fill_value='extrapolate')
		self._atten_corr = atten_corr
		return atten_corr

		
	def geometry_correction(self, distance, r_det, thickness, sample_size, shape='circle', N=1E6):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		N = int(N)

		if shape.lower()=='circle':
			r = np.sqrt(np.random.uniform(size=N))*sample_size
			ang = np.pi*2.0*np.random.uniform(size=N)
			x0 = r*np.cos(ang)
			y0 = r*np.sin(ang)

		elif shape.lower()=='square':
			x0 = (np.random.uniform(size=N)-0.5)*sample_size
			y0 = (np.random.uniform(size=N)-0.5)*sample_size

		elif shape.lower()=='rectangle':
			x0 = (np.random.uniform(size=N)-0.5)*sample_size[0]
			y0 = (np.random.uniform(size=N)-0.5)*sample_size[1]

		tht = np.arccos(2.0*np.random.uniform(size=N)-1.0)
		phi = 2.0*np.pi*np.random.uniform(size=N)

		z = distance+np.random.uniform(size=N)*thickness
		xf = x0 + z*np.tan(tht)*np.cos(phi)
		yf = y0 + z*np.tan(tht)*np.sin(phi)

		N_i = float(len(np.where(np.sqrt(xf**2+yf**2)<=r_det)[0]))
		if N_i<1E4:
			print('WARNING: Uncertainty in solid-angle {}% -- Increase N'.format(round(1E2*np.sqrt(N_i)/N_i, 1)))

		SA = 2.0*np.pi*N_i/float(N)
		corr = SA/(2.0*np.pi*(1.0-distance/np.sqrt(distance**2+r_det**2)))
		self._geom_corr = corr

		return corr



	def _snip_bg(self):
		adj = self.fit_config['snip_adj']

		x, dead = np.arange(len(self.hist)), int(7.5*adj*self.cb.res(len(self.hist)))
		V_i, L = np.log(np.log(np.sqrt(self.hist+1.0)+1.0)+1.0), len(x)-dead
		while self.hist[dead]==0:
			dead += 1
		
		for M in np.linspace(0, 7.5*adj, 10):
			l, h = np.array(x-M*self.cb.res(x), dtype=np.int32), np.array(x+M*self.cb.res(x), dtype=np.int32)
			V_i[dead:L] = np.minimum(V_i[dead:L], 0.5*(V_i[l[dead:L]]+V_i[h[dead:L]]))

		snip = (np.exp(np.exp(V_i)-1.0)-1.0)**2-1.0
		snip += adj*1.5*np.sqrt(snip+0.01)

		a1, a2 = 0.3/(adj*self.cb.res(0.1*L)), 0.3/(adj*self.cb.res(0.9*L))
		wt1, wt2 = x[::-1]/x[-1], x/x[-1]

		def exp_smooth(x, alpha):
			N = int(2.0/alpha)-1
			wts = np.array([alpha*(1.0-alpha)**abs(N-i) for i in range(2*N+1)])
			y = np.concatenate((np.ones(N)*x[0], x, np.ones(N)*x[-1]))
			return np.dot(wts/np.sum(wts), np.take(y, [np.arange(i,len(x)+i) for i in range(2*N+1)]))

		return wt1*exp_smooth(snip, a1) + wt2*exp_smooth(snip, a2)


	def _forward_fit(self, engcal=None):
		itps, gm = self._gammas()

		L = len(self.hist)
		chan = np.arange(L)
		W = self.fit_config['pk_width']
		R, alpha, step = self.fit_config['R'], self.fit_config['alpha'], self.fit_config['step']
		X = np.zeros((len(itps), L))

		for n,ip in enumerate(itps):
			idx = self.cb.map_channel(gm[n]['energy'], engcal)
			sig = self.cb.res(idx)
			A_norm = self.cb.eff(gm[n]['energy'])*gm[n]['intensity']/sig
			whr = np.where(((idx+W*sig)<L)&((idx-W*sig)>0))[0]
			idx, sig, A_norm = idx[whr], sig[whr], A_norm[whr]

			l, h = np.array(idx-W*sig, dtype=np.int32), np.array(idx+W*sig, dtype=np.int32)
			for m,i in enumerate(idx):
				X[n][l[m]:h[m]] += self._peak(chan[l[m]:h[m]], A_norm[m], i, sig[m], R, alpha, step)

		Y = self.hist-self._snip
		return X, Y, np.dot(np.linalg.inv(np.dot(X, X.T)), np.dot(X, Y.T).T)

	def auto_calibrate(self, guess=None, data=None):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		from scipy.optimize import differential_evolution

		if guess is None:
			guess = list(self.cb.engcal)

		if data is not None:
			data = np.asarray(data, dtype=np.float64)
			if len(data)==1:
				guess = [0.0, data[0][1]/data[0][0]]
			else:
				N = 2 if len(data)<5 else 3
				M = np.column_stack([data[:,0]**m for m in range(N)])
				b = np.array([np.sum(data[:,0]**m*data[:,1]) for m in range(N)])
				M_inv = np.linalg.inv(np.dot(M.T, M))
				guess = np.dot(M_inv, b).tolist()

		guess = guess if len(guess)==3 else list(guess)+[0.0]
		
		def obj(*m):
			X, Y, B = self._forward_fit(engcal=[guess[0], m[0], guess[2]])
			return np.sum(np.sqrt(np.abs(Y-np.dot(B, X))))

		self.cb.engcal = [guess[0], differential_evolution(obj, [(0.995*guess[1], 1.005*guess[1])]).x[0], guess[2]]


	def _gammas(self):
		if self._gmls is None:
			itps, gm = [], []
			for i in self.isotopes:
				g = Isotope(i).gammas(I_lim=self.fit_config['I_min'], E_lim=self.fit_config['E_min'], dE_511=self.fit_config['dE_511'])
				g['intensity'] = g['intensity']*1E-2
				g['unc_intensity'] = g['unc_intensity']*1E-2
				if len(g):
					itps.append(i)
					gm.append(g)
			self._gmls = (itps, gm)
			return itps, gm
		else:
			return self._gmls

	def _chi2(self, fit, l ,h):
		non_zero = np.where(self.hist[l:h]>0)
		dof = float(len(non_zero[0])-len(fit)-1)
		if dof==0:
			return np.inf
		resid = self.hist[l:h][non_zero]-self._multiplet(np.arange(l,h)[non_zero], *fit)
		return np.sum(resid**2/self.hist[l:h][non_zero])/dof
		
	def _unc_calc(self, fn, p0, cov):
		var, eps = 0.0, 1E-8
		for n in range(len(p0)):
			for m in range(n, len(p0)):
				c_n, c_m = list(p0), list(p0)
				c_n[n], c_m[m] = c_n[n]+eps, c_m[m]+eps
				par_n = (fn(*c_n)-fn(*p0))/eps
				par_m = (fn(*c_m)-fn(*p0))/eps
				var += cov[n][m]*par_n*par_m*(2.0 if n!=m else 1.0)
		return var
	
	def _counts(self, fit, cov):
		cfg = self.fit_config
		L = 3+2*int(cfg['skew_fit'])+int(cfg['step_fit'])
		M = {'snip':0, 'constant':1, 'linear':2, 'quadratic':3}[cfg['bg'].lower()]
		N_cts, unc_N = [], []

		skew_fn = lambda A, R, alpha, sig: 2*A*R*alpha*sig*np.exp(-0.5/alpha**2)
		min_skew_fn = lambda A, sig: 2*A*cfg['R']*cfg['alpha']*sig*np.exp(-0.5/cfg['alpha']**2)
		pk_fn = lambda A, sig: 2.506628*A*sig

		for m in range(int((len(fit)-M)/L)):
			i = L*m+M
			p_i = [i, i+2]
			s_i = [i, i+3, i+4, i+2]
			if cfg['skew_fit']:
				N_cts.append(pk_fn(*fit[p_i])+skew_fn(*fit[s_i]))
			else:
				N_cts.append(pk_fn(*fit[p_i])+min_skew_fn(*fit[p_i]))
			if cov is not None:
				if not np.isinf(cov[i][i]):
					if cfg['skew_fit']:
						skew_unc = self._unc_calc(skew_fn, fit[s_i], cov[np.ix_(s_i, s_i)])
					else:
						skew_unc = self._unc_calc(min_skew_fn, fit[p_i], cov[np.ix_(p_i, p_i)])
					unc_N.append(np.sqrt(self._unc_calc(pk_fn, fit[p_i], cov[np.ix_(p_i, p_i)])+skew_unc))
				else:
					unc_N.append(np.inf)
			else:
				unc_N.append(np.inf)
		return np.array(N_cts), np.array(unc_N)
	
	def _decays(self, N, unc_N, df):
		if self._atten_corr is None:
			corr = self._geom_corr
		else:
			corr = self._geom_corr*self._atten_corr(df['energy'])
		D = N/(df['intensity']*self.cb.eff(df['energy'])*corr*(self.live_time/self.real_time))
		unc_D = np.sqrt((N/N**2)+(unc_N/N)**2+(self.cb.unc_eff(df['energy'])/self.cb.eff(df['energy']))**2+(df['unc_intensity']/df['intensity'])**2)*D

		A, unc_A = D/self.real_time, unc_D/self.real_time
		return D, unc_D, A, unc_A

	def _peak(self, x, A, mu, sig, R, alpha, step):
		r2 = 1.41421356237
		return A*np.exp(-0.5*((x-mu)/sig)**2)+R*A*np.exp((x-mu)/(alpha*sig))*erfc((x-mu)/(r2*sig)+1.0/(r2*alpha))+step*A*erfc((x-mu)/(r2*sig))
		
	def _multiplet(self, x, *args):
		bg_fit = self.fit_config['bg'].lower()
		if bg_fit=='snip':
			b, peak = 0, self._snip_interp(x)
		elif bg_fit=='constant':
			b, peak = 1, args[0]*np.ones(len(x))
		elif bg_fit=='linear':
			b, peak = 2, args[0]+args[1]*x
		elif bg_fit=='quadratic':
			b, peak = 3, args[0]+args[1]*x+args[2]*x**2

		R, alpha, step = self.fit_config['R'], self.fit_config['alpha'], self.fit_config['step']
		if self.fit_config['skew_fit']:
			if self.fit_config['step_fit']:
				for n in range(int((len(args)-b)/6)):
					peak += self._peak(x,*args[6*n+b:6*n+6+b])
			else:
				for n in range(int((len(args)-b)/5)):
					peak += self._peak(x,*(args[5*n+b:5*n+5+b]+(step,)))
		else:
			if self.fit_config['step_fit']:
				for n in range(int((len(args)-b)/4)):
					peak += self._peak(x,*(args[4*n+b:4*n+b+3]+(R, alpha)+(args[4*n+3+b])))
			else:
				for n in range(int((len(args)-b)/3)):
					peak += self._peak(x,*(args[3*n+b:3*n+b+3]+(R,alpha,step)))
		return peak

	def _get_p0(self, gammas=None):
		### gammas must at least have energy, intensity, unc_intensity (isotope optional)
		istp, gm = self._gammas()
		for n,i in enumerate(istp):
			gm[n]['isotope'] = i
		if gammas is not None:
			gm += [pd.DataFrame(gammas)]
			gm[-1]['intensity'] = gm[-1]['intensity']*1E-2
			gm[-1]['unc_intensity'] = gm[-1]['unc_intensity']*1E-2

		X, Y, B = self._forward_fit()
		L = len(self.hist)
		chan = np.arange(L)

		df = pd.concat(gm, sort=True, ignore_index=True).sort_values(by=['energy']).reset_index(drop=True)
		df['idx'] = self.cb.map_channel(df['energy'])
		df['sig'] = self.cb.res(df['idx'])
		df['l'] = np.array(df['idx']-self.fit_config['pk_width']*df['sig'], dtype=np.int32)
		df['h'] = np.array(df['idx']+self.fit_config['pk_width']*df['sig'], dtype=np.int32)
		df['A'] = (self.hist-self._snip)[df['idx']]

		for n,i in enumerate(istp):
			df_sub = df[df['isotope']==i]
			df.loc[df_sub.index, 'A'] = B[n]*self.cb.eff(df_sub['energy'])*df_sub['intensity']/df_sub['sig']

		df['SNR'] = df['A']/np.sqrt(self._snip[df['idx']])
		df = df[(df['SNR']>self.fit_config['SNR_min'])&(df['l']>0)&(df['h']<L)].reset_index(drop=True)

		if len(df)==0:
			return []

		multiplets = [df.loc[[0]]]
		for n,l in enumerate(df['l'].to_numpy()):
			if n>0:
				if l<multiplets[-1].loc[n-1, 'h'] and len(multiplets[-1])<self.fit_config['multi_max']:
					multiplets[-1] = pd.concat([multiplets[-1], df.loc[[n]]])
				else:
					multiplets.append(df.loc[[n]])

		p0 = []
		for multi in multiplets:
			p = {'df':multi, 'l':multi['l'].min(), 'h':multi['h'].max(), 'p0':[], 'bounds':[[],[]], 'istp':multi['isotope'].to_list()}

			bgf = self.fit_config['bg'].lower()
			if bgf=='constant':
				p['p0'].append(np.average(self._snip[p['l']:p['h']]))
				p['bounds'][0].append(0.85*p['p0'][0])
				p['bounds'][1].append(1.15*p['p0'][0])

			elif bgf=='linear' or bgf=='quadratic':
				N = 2 if bgf=='linear' else 3
				x = chan[p['l']:p['h']]
				M = np.column_stack([x**m for m in range(N)])
				b = np.array([np.sum(x**m*self._snip[p['l']:p['h']]) for m in range(N)])
				M_inv = np.linalg.inv(np.dot(M.T, M))
				p['p0'] += np.dot(M_inv, b).tolist()
				resid = self._snip[p['l']:p['h']]-np.dot(M, p['p0'])
				unc = np.sqrt(np.abs(M_inv*np.dot(resid.T, resid)/max([float((p['h']-p['l'])-N), 1.0])))
				p['bounds'][0] += [i-100*unc[n][n] for n,i in enumerate(p['p0'])]
				p['bounds'][1] += [i+100*unc[n][n] for n,i in enumerate(p['p0'])]

			R, alpha, step = self.fit_config['R'], self.fit_config['alpha'], self.fit_config['step']
			bA, bm, bs = self.fit_config['A_max'], self.fit_config['mu_max'], self.fit_config['sig_max']
			for n,rw in multi.iterrows():
				p['p0'] += [rw['A'], rw['idx'], rw['sig']]
				p['bounds'][0] += [0.0, rw['idx']-bm*rw['sig'], rw['sig']/bs]
				p['bounds'][1] += [rw['A']*bA, rw['idx']+bm*rw['sig'], rw['sig']*bs]

				if self.fit_config['skew_fit']:
					p['p0'] += [R, alpha]
					p['bounds'][0] += [0.0,0.5]
					p['bounds'][1] += [1.0, max((2.5, alpha))]

				if self.fit_config['skew_fit']:
					p['p0'] += [step]
					p['bounds'][0] += [0.0]
					p['bounds'][1] += [0.1]

			p0.append(p)

		return p0

	def _split_fits(self):
		fits, istp, lh, N_sub = [], [], [], []
		cfg = self.fit_config
		if self._fits is None:
			self.fit_peaks()

		for ft in self._fits:
			f = ft['fit']
			B = {'snip':0, 'constant':1, 'linear':2, 'quadratic':3}[cfg['bg'].lower()]
			p0 = f[:B].tolist()
			L = 3+2*int(cfg['skew_fit'])+int(cfg['step_fit'])
			N_sub.append(int((len(f)-B)/L))
			for n in range(N_sub[-1]):
				if type(ft['istp'][n])==str:
					istp.append(ft['istp'][n])
				else:
					istp.append(None)
				fits.append(p0+f[B+n*L:B+(n+1)*L].tolist())
				mu, sig = f[B+n*L+1], f[B+n*L+2]
				lh.append([mu-cfg['pk_width']*sig, mu+cfg['pk_width']*sig])
		itp_set = sorted(list(set(istp)))
		return {i:[[fits[n],lh[n]] for n,ip in enumerate(istp) if ip==i] for i in itp_set}, itp_set, N_sub


	def _multi_fit(self, p0):
		chan = np.arange(len(self.hist))
		try:
			fit, unc = curve_fit(self._multiplet, chan[p0['l']:p0['h']], self.hist[p0['l']:p0['h']], p0=p0['p0'], bounds=p0['bounds'], sigma=np.sqrt(self.hist[p0['l']:p0['h']]+0.1))
			p0['fit'] = fit
			p0['unc'] = unc
			N, unc_N = self._counts(fit, unc)
			D, unc_D, A, unc_A = self._decays(N, unc_N, p0['df'])
			chi2 = self._chi2(fit, p0['l'], p0['h'])

			f = p0['df']
			cols = ['filename','isotope','energy','counts','unc_counts',
				'intensity','unc_intensity','efficiency','unc_efficiency',
				'decays','unc_decays','decay_rate','unc_decay_rate','chi2',
				'start_time', 'live_time', 'real_time']

			df = pd.DataFrame({'filename':self.filename, 'isotope':f['isotope'], 'energy':f['energy'],
							'counts':N, 'unc_counts':unc_N, 'intensity':f['intensity'],
							'unc_intensity':f['unc_intensity'], 'efficiency':self.cb.eff(f['energy']),
							'unc_efficiency':self.cb.unc_eff(f['energy']), 'decays':D,
							'unc_decays':unc_D, 'decay_rate':A, 'unc_decay_rate':unc_A, 'chi2':chi2,
							'start_time':dtm.datetime.strftime(self.start_time, '%m/%d/%Y %H:%M:%S'),
							'live_time':self.live_time, 'real_time':self.real_time}, columns=cols)

			return p0, df
		except:
			return p0, p0['df']

	def fit_peaks(self, gammas=None, **kwargs):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		self.fit_config = kwargs
		p0 = self._get_p0(gammas)
		if len(p0):
			multiplets = list(map(self._multi_fit, p0))
			self._fits = [i[0] for i in multiplets if 'fit' in i[0]]
			self._peaks = pd.concat([i[1] for i in multiplets if 'fit' in i[0]], ignore_index=True)
		else:
			self._fits = []
			self._peaks = None

		return self._peaks


	@property
	def peaks(self):
		if self._peaks is None:
			self._peaks = self.fit_peaks()
		return self._peaks
	
		
	def saveas(self, filename, replace=False):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		if any([filename.endswith(e) for e in ['.png','.pdf','.eps','.pgf','.ps','.raw','.rgba','.svg','.svgz']]):
				self.plot(saveas=filename, show=False)

		if filename.endswith('.Spe'):
			### Maestro ASCII .Spe ###
			self._ortec_metadata['DATE_MEA'] = [dtm.datetime.strftime(self.start_time, '%m/%d/%Y %H:%M:%S')]
			self._ortec_metadata['MEAS_TIM'] = ['{0} {1}'.format(int(self.live_time), int(self.real_time))]

			self._ortec_metadata['ENER_FIT'] = ['{0} {1}'.format(self.cb.engcal[0], self.cb.engcal[1])]
			self._ortec_metadata['MCA_CAL'] = ['3','{0} {1} {2} keV'.format(self.cb.engcal[0], self.cb.engcal[1], (self.cb.engcal[2] if len(self.cb.engcal)>2 else 0.0))]

			defaults = {'ROI':['0'],'SPEC_REM':['DET# 0','DETDESC# None','AP# Maestro Version 7.01'],'PRESETS':['0'],
						'SHAPE_CAL':['3','0E+00 0E+00 0E+00'],'SPEC_ID':['No sample description was entered.']}

			for d in defaults:
				if d not in self._ortec_metadata:
					self._ortec_metadata[d] = defaults[d]

			ss = '\n'.join(['${}:\n'.format(sc)+'\n'.join(self._ortec_metadata[sc]) for sc in ['SPEC_ID','SPEC_REM','DATE_MEA','MEAS_TIM']])
			ss += '\n$DATA:\n0 {}\n'.format(len(self.hist)-1)
			ss += '\n'.join([' '*(8-len(i))+i for i in map(str, self.hist)])+'\n'
			ss += '\n'.join(['${}:\n'.format(sc)+'\n'.join(self._ortec_metadata[sc]) for sc in ['ROI','PRESETS','ENER_FIT','MCA_CAL','SHAPE_CAL']])+'\n'

			with open(filename,'w') as f:
				f.write(ss)

		if filename.endswith('.Chn'):
			### Maestro integer .Chn ###
			if 'SPEC_REM' in self._ortec_metadata:
				det_no = int(self._ortec_metadata['SPEC_REM'][0].split(' ')[1].strip())
			else:
				det_no = 0

			ss = np.array([-1, det_no, 1], dtype='i2').tobytes()

			st = self.start_time
			ss += np.array(('0' if st.second<10 else '')+str(st.second), dtype='S2').tobytes()
			ss += np.array([self.real_time/0.02, self.live_time/0.02], dtype='i4').tobytes()
			months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
			ss += np.array(('0' if st.day<10 else '')+str(st.day)+months[st.month]+str(st.year)[-2:]+('1' if st.year>1999 else '0'), dtype='S8').tobytes()
			ss += np.array(('0' if st.hour<10 else '')+str(st.hour)+('0' if st.minute<10 else '')+str(st.minute), dtype='S4').tobytes()

			ss += np.array([0, len(self.hist)], dtype='i2').tobytes()
			ss += np.array(self.hist, dtype='i4').tobytes()
			ss += np.array([-102, 0], dtype='i2').tobytes()

			ss += np.array(self.cb.engcal, dtype='f4').tobytes()

			if 'SHAPE_CAL' in self._ortec_metadata:
				ss += np.array(self._ortec_metadata['SHAPE_CAL'][1].split(' '), dtype='f4').tobytes()
			else:
				ss += np.array([0,0,0], dtype='f4').tobytes()
			ss += np.zeros(228, dtype='i1').tobytes()

			if 'SPEC_REM' in self._ortec_metadata:
				L = min((63, len(self._ortec_metadata['SPEC_REM'][1].split('# ')[1])))
				ss += np.array(L, dtype='i1').tobytes()
				ss += np.array(self._ortec_metadata['SPEC_REM'][1].split('# ')[1][:L], dtype='S{}'.format(L)).tobytes()
			else:
				ss += np.array(0, dtype='i1').tobytes()

			if 'SPEC_ID' in self._ortec_metadata:
				if self._ortec_metadata['SPEC_ID'][0]!='No sample description was entered.':
					L = len(''.join(self._ortec_metadata['SPEC_ID']))
					L = min((63, L))
					ss += np.array(L, dtype='i1').tobytes()
					ss += np.array(''.join(self._ortec_metadata['SPEC_ID'])[:L])
				else:
					ss += np.array(0, dtype='i1').tobytes()
					ss += np.array('\x00'*63, dtype='S63').tobytes()
			else:
				ss += np.array(0, dtype='i1').tobytes()
				ss += np.array('\x00'*63, dtype='S63').tobytes()

			ss += np.array('\x00'*128, dtype='S128').tobytes()
			with open(filename, 'wb') as f:
				f.write(ss)


		if filename.endswith('.spe'):
			### Radware gf3 .spe ###
			name = self.filename[:8]+' '*(8-len(self.filename))
			with open(filename, 'wb') as f:
				ss = np.array(24, dtype=np.uint32).tobytes()
				ss += np.array(name, dtype='c').tobytes()
				ss += np.array([len(self.hist), 1, 1, 1, 24, 4*len(self.hist)], dtype=np.uint32).tobytes()
				ss += np.array(self.hist, dtype=np.float32).tobytes()
				ss += np.array(4*len(self.hist), dtype=np.uint32).tobytes()
				f.write(ss)

		if filename.endswith('.csv'):
			if os.path.exists(filename) and not replace:
				df = pd.read_csv(filename, header=0)
				df = df[df['filename']!=self.filename]
				df = pd.concat([df, self.peaks])
				df.to_csv(filename, index=False)
			else:
				self.peaks.to_csv(filename, index=False)

		if filename.endswith('.db'):
			if os.path.exists(filename) and not replace:
				con = _get_connection(filename)
				df = pd.read_sql('SELECT * FROM peaks', con)
				df = df[df['filename']!=self.filename]
				df = pd.concat([df, self.peaks])
				df.to_sql('peaks', con, if_exists='replace', index=False)
			else:
				self.peaks.to_sql('peaks', _get_connection(filename), if_exists='replace', index=False)

		if filename.endswith('.json'):
			if os.path.exists(filename) and not replace:
				df = pd.read_json(filename, orient='records')
				df = df[df['filename']!=self.filename][self.peaks.columns]
				df = pd.concat([df, self.peaks])
				json.dump(json.loads(df.to_json(orient='records')), open(filename, 'w'), indent=4)
			else:
				json.dump(json.loads(self.peaks.to_json(orient='records')), open(filename, 'w'), indent=4)

		
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

		for n,p in self.peaks.iterrows():
			ln1 = [p['energy'], p['isotope'], p['intensity']*100.0]
			ln0 = '{1} - {0} keV (I = {2}%)'.format(*ln1)
			ss = ln0 + '\n'
			ss += ''.join(['-']*len(ln0)) + '\n'
			ln2 = [int(p['counts']), (int(p['unc_counts']) if np.isfinite(p['unc_counts']) else np.inf),
					format(p['decays'], '.3e'), format(p['unc_decays'], '.3e'), 
					format(p['decay_rate'], '.3e'), format(p['unc_decay_rate'], '.3e'), 
					format(p['decay_rate']/3.7E4, '.3e'), format(p['unc_decay_rate']/3.7E4, '.3e'), 
					round(p['chi2'], 3)]
			ss += 'counts: {0} +/- {1}'.format(ln2[0], ln2[1]) + '\n'
			ss += 'decays: {0} +/- {1}'.format(ln2[2], ln2[3]) + '\n'
			ss += 'activity (Bq): {0} +/- {1}'.format(ln2[4], ln2[5]) + '\n'
			ss += 'activity (uCi): {0} +/- {1}'.format(ln2[6], ln2[7]) + '\n'
			ss += 'chi2/dof: {}'.format(ln2[8]) + '\n'
			print(ss+'\n')

		
	def plot(self, fit=True, xcalib=True, **kwargs):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		f, ax = _init_plot(figsize=(12.8, 4.8), **kwargs)

		chan = np.arange(len(self.hist))
		xgrid = np.array([chan-0.5, chan+0.5]).T.flatten()
		spec = np.array([self.hist, self.hist]).T.flatten()
		erange = self.cb.eng(xgrid) if xcalib else xgrid

		spec_label = self.filename.split('/')[-1] if self.filename is not None else None
		ax.plot(erange, spec, lw=1.2, zorder=1, label=spec_label)

		lbs = []
		if fit:
			cm, cl = colormap(), colormap(aslist=True)
			sub, itp, N_sub = self._split_fits()
			
			for n,p in enumerate(self._fits):
				if N_sub[n]>1:
					xgrid = np.arange(p['l'], p['h'], 0.1)
					pk_fit = self._multiplet(xgrid, *p['fit'])
					erange = self.cb.eng(xgrid) if xcalib else xgrid
					ax.plot(erange, np.where(pk_fit>0.1, pk_fit, 0.1), ls='--', lw=1.4, color=cm['gy'], alpha=0.8)

			
			cl = [c for c in cl if c not in [cm['k'], cm['gy']]]
			ls = ['-','-.','--']

			for n,i in enumerate(itp):
				c = cl[n%len(cl)]
				ilbl = Isotope(i).TeX if type(i)==str else None

				for p,lh in sub[i]:
					xgrid = np.arange(lh[0], lh[1], 0.1)
					erange = self.cb.eng(xgrid) if xcalib else xgrid
					pk_fit = self._multiplet(xgrid, *p)

					lb = ilbl if (ilbl not in lbs and len(lbs)<20) else None
					if lb is not None:
						lbs.append(lb)

					ax.plot(erange, np.where(pk_fit>0.1, pk_fit, 0.1), lw=1.4, color=c, label=lb, ls=ls[int(n/len(cl))%len(ls)])
					
		
		if xcalib:
			ax.set_xlabel('Energy (keV)')
		else:
			ax.set_xlabel('ADC Channel')
		ax.set_ylabel('Counts')
		
		ax.legend(loc=0, ncol=min((max((int(len(lbs)/5), 1)), 3)))
		kwargs['_default_log'] = True

		return _draw_plot(f, ax, **kwargs)
		