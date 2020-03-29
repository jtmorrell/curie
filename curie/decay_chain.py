from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import datetime as dtm

from scipy.optimize import curve_fit

from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from .isotope import Isotope
from .spectrum import Spectrum

class DecayChain(object):
	"""DecayChain

	...
	
	Parameters
	----------
	parent_isotope : str
		Description of parameter x

	R : str, array_like, dict or pd.DataFrame
		Description of parameter x

	A0 : float or dict
		Description of parameter x

	units : str, optional
		Description of parameter x

	Attributes
	----------
	isotopes : list
		Description

	counts : pd.DataFrame
		Description

	R_avg : pd.DataFrame
		Description


	Examples
	--------

	"""

	def __init__(self, parent_isotope, R=None, A0=None, units='s'):
		if units.lower() in ['hr','min','yr','sec']:
			units = {'hr':'h','min':'m','yr':'y','sec':'s'}[units.lower()]
		self.units = units
		

		istps = [Isotope(parent_isotope)]
		self.isotopes = [istps[0].name]
		self.R, self.A0, self._counts = None, {self.isotopes[0]:0.0}, None
		self.chain = [[istps[0].decay_const(units), [], []]]
		stable_chain = [False]

		while not all(stable_chain):
			for n in [n for n,ch in enumerate(stable_chain) if not ch]:
				stable_chain[n] = True
				for prod in istps[n].decay_products:
					br = istps[n].decay_products[prod]
					I = Isotope(prod)
					if I.name in self.isotopes:
						self.chain[self.isotopes.index(I.name)][1].append(br)
						self.chain[self.isotopes.index(I.name)][2].append(n)
					else:
						istps.append(I)
						self.isotopes.append(I.name)
						self.chain.append([I.decay_const(units), [br], [n]])
						stable_chain.append(self.chain[-1][0]<1E-12)
						if not stable_chain[-1]:
							self.A0[self.isotopes[-1]] = 0.0
		self.chain = np.array(self.chain, dtype=object)


		if A0 is not None:
			if type(A0)==float or type(A0)==int:
				self.A0 = {self.isotopes[0]:float(A0)}
			else:
				self.A0 = {self._filter_name(i):float(A0[i]) for i in A0}


		if R is not None:
			if type(R)==str:
				if R.endswith('.json'):
					self.R = pd.DataFrame(json.loads(open(R).read()))
				elif R.endswith('.csv'):
					self.R = pd.read_csv(R, header=0).fillna(method='ffill')
				if 'isotope' not in self.R.columns.to_list():
					self.R['isotope'] = self.isotopes[0]

			elif type(R)==dict:
				self.R = pd.DataFrame(R)

			elif type(R)==list or type(R)==np.ndarray:
				R = np.asarray(R)
				self.R = pd.DataFrame({'isotope':self.isotopes[0], 'R':R[:,0], 'time':R[:,1]})

			self.R['isotope'] = [self._filter_name(i) for i in self.R['isotope']]
			# self.A0 = {p:0.0 for p in self.isotopes}

			time = np.insert(np.unique(self.R['time']), 0, [0.0])
			for n,dt in enumerate(time[1:]-time[:-1]):
				_R_dict = {p:self.R[self.R['isotope']==p].iloc[n]['R'] for p in pd.unique(self.R['isotope'])}
				self.A0 = {p:self.activity(p, dt, _R_dict=_R_dict) for p in self.A0}

	def __str__(self):
		return self.isotopes[0]

	def _filter_name(self, istp):
		return Isotope(istp).name

	def _index(self, istp):
		return self.isotopes.index(self._filter_name(istp))

	def _get_branches(self, istp):
		if self._filter_name(istp) not in self.isotopes:
			return [], []

		m = self._index(istp)
		BR = [[0.0]]+[[r] for r in self.chain[m,1]]
		CH = [[m]]+[[n] for n in self.chain[m,2]]

		while not all([c[-1]==0 for c in CH]):
			BR = [BR[n]+[i] for n,c in enumerate(CH) for i in self.chain[c[-1],1]]
			CH = [c+[i] for c in CH for i in self.chain[c[-1],2]]

		return [np.array(r)[::-1] for r in BR], [np.array(c)[::-1] for c in CH]

	def _r_lm(self, units=None, r_half_conv=False):
		if units is None:
			return 1.0

		if units.lower() in ['hr','min','yr','sec']:
			units = {'hr':'h','min':'m','yr':'y','sec':'s'}[units.lower()]

		half_conv = {'ns':1E-9, 'us':1E-6, 'ms':1E-3,
					's':1.0, 'm':60.0, 'h':3600.0,
					'd':86400.0, 'y':31557.6E3, 'ky':31557.6E6,
					'My':31557.6E9, 'Gy':31557.6E12}

		if r_half_conv:
			return half_conv[units]

		return half_conv[units]/half_conv[self.units]


	def activity(self, isotope, time, units=None, _R_dict=None, _A_dict=None):
		""" Description

		...

		Parameters
		----------
		isotope : str
			Description of x

		time : array_like
			Description

		units : str, optional
			Description

		Returns
		-------
		np.ndarray
			Description

		Examples
		--------

		"""

		time = np.asarray(time)
		A = np.zeros(len(time)) if time.shape else np.array(0.0)

		for m,(BR, chain) in enumerate(zip(*self._get_branches(isotope))):
			lm = self._r_lm(units)*self.chain[chain, 0]
			L = len(chain)
			for i in range(L):
				if i==L-1 and m>0:
					continue

				ip = self.isotopes[chain[i]]
				A0 = self.A0[ip] if _A_dict is None else _A_dict[ip]
				A_i = lm[-1]*(A0/lm[i])
				
				B_i = np.prod(lm[i:-1]*BR[i:-1])

				for j in range(i, L):
					K = np.arange(i, L)
					d_lm = lm[K[K!=j]]-lm[j]
					C_j = np.prod(np.where(np.abs(d_lm)>1E-12, d_lm, 1E-12*np.sign(d_lm)))
					A += A_i*B_i*np.exp(-lm[j]*time)/C_j
					if _R_dict is not None:
						if ip in _R_dict:
							if lm[j]>1E-12:
								A += _R_dict[ip]*lm[-1]*B_i*(1.0-np.exp(-lm[j]*time))/(lm[j]*C_j)
							else:
								A += _R_dict[ip]*lm[-1]*B_i*time/C_j
		return A
		
	def decays(self, isotope, t_start, t_stop, units=None, _A_dict=None):
		""" Description

		...

		Parameters
		----------
		isotope : str
			Description of x

		t_start : array_like
			Description

		t_stop : array_like
			Description

		units : str, optional
			Description

		Returns
		-------
		np.ndarray
			Description

		Examples
		--------

		"""

		t_start, t_stop = np.asarray(t_start), np.asarray(t_stop)
		D = np.zeros(len(t_start)) if t_start.shape else (np.zeros(len(t_stop)) if t_stop.shape else np.array(0.0))

		for m,(BR, chain) in enumerate(zip(*self._get_branches(isotope))):
			lm = self._r_lm(units)*self.chain[chain,0]
			L = len(chain)
			for i in range(L):
				if i==L-1 and m>0:
					continue

				ip = self.isotopes[chain[i]]
				A0 = self.A0[ip] if _A_dict is None else _A_dict[ip]
				A_i = lm[-1]*(self.A0[self.isotopes[chain[i]]]/lm[i])
				B_i = np.prod(lm[i:-1]*BR[i:-1])

				for j in range(i, len(chain)):
					K = np.arange(i, len(chain))
					d_lm = lm[K[K!=j]]-lm[j]
					C_j = np.prod(np.where(np.abs(d_lm)>1E-12, d_lm, 1E-12*np.sign(d_lm)))
					if lm[j]>1E-12:
						D += A_i*B_i*(np.exp(-lm[j]*t_start)-np.exp(-lm[j]*t_stop))/(lm[j]*C_j)
					else:
						D += A_i*B_i*(t_stop-t_start)/C_j

		return D*self._r_lm((self.units if units is None else units), True)

	@property
	def counts(self):
		return self._counts

	@counts.setter
	def counts(self, N_c):
		if N_c is not None:
			if type(N_c)==pd.DataFrame:
				self._counts = N_c

			elif type(N_c)!=dict:
				N_c = np.asarray(N_c)
				self._counts = pd.DataFrame({'isotope':self.isotopes[0],
											'start':N_c[:,0],
											'stop':N_c[:,1],
											'counts':N_c[:,2],
											'unc_counts':N_c[:,3]})

			else:
				self._counts = pd.DataFrame(N_c)
				self._counts['isotope'] = [self._filter_name(i) for i in self._counts['isotope']]
			
		
	def get_counts(self, spectra, EoB, peak_data=None):
		""" Description

		...

		Parameters
		----------
		spectra : list
			Description of x
		
		EoB : str or datetime.datetime
			Description

		peak_data : str or pd.DataFrame, optional
			Description

		Examples
		--------

		"""

		counts = []
		if type(EoB)==str:
			EoB = dtm.datetime.strptime(EoB, '%m/%d/%Y %H:%M:%S')

		if peak_data is not None:
			if type(peak_data)==str:
				if peak_data.endswith('.json'):
					peak_data = pd.read_json(peak_data, orient='records')
				elif peak_data.endswith('.csv'):
					peak_data = pd.read_csv(peak_data, header=0)
				elif peak_data.endswith('.db'):
					peak_data = pd.read_sql('SELECT * FROM peaks', _get_connection(peak_data))

			else:
				peak_data = pd.DataFrame(peak_data)

		for sp in spectra:
			if type(sp)==str:
				if peak_data is not None:
					df = peak_data[peak_data['filename']==sp]
					df['isotope'] = [self._filter_name(i) for i in df['isotope']]
					df = df[df['isotope'].isin(self.isotopes)]

					if len(df):
						start_time = df.iloc[0]['start_time']
						if type(start_time)==str:
							start_time = dtm.datetime.strptime(start_time, '%m/%d/%Y %H:%M:%S')
						start = (start_time-EoB).total_seconds()*self._r_lm('s')
						stop = start+(df.iloc[0]['real_time']*self._r_lm('s'))
				else:
					raise ValueError('peak_data must be specified if type(spectra)==str')
			else:
				if peak_data is not None:
					df = peak_data[peak_data['filename']==sp.filename]
				else:
					df = sp.peaks.copy()

				df['isotope'] = [self._filter_name(i) for i in df['isotope']]
				df = df[df['isotope'].isin(self.isotopes)]

				if len(df):
					start = (sp.start_time-EoB).total_seconds()*self._r_lm('s')
					stop = start+(sp.real_time*self._r_lm('s'))

			counts.append(pd.DataFrame({'isotope':df['isotope'], 'start':start, 'stop':stop, 'counts':df['decays'], 'unc_counts':df['unc_decays']}))

		self.counts = pd.concat(counts, sort=True, ignore_index=True).sort_values(by=['start']).reset_index(drop=True)
		self.counts['activity'] = [p['counts']*self.activity(p['isotope'], p['start'])/self.decays(p['isotope'], p['start'], p['stop']) for n,p in self.counts.iterrows()]
		self.counts['unc_activity'] = self.counts['unc_counts']*self.counts['activity']/self.counts['counts']


	@property	
	def R_avg(self):
		df = []
		for ip in np.unique(self.R['isotope']):
			time = np.insert(np.unique(self.R['time']), 0, [0.0])
			df.append({'isotope':ip, 'R_avg':np.average(self.R[self.R['isotope']==ip]['R'], weights=time[1:]-time[:-1])})
		return pd.DataFrame(df)
		
	def fit_R(self):
		""" Description

		...

		Returns
		-------
		list
			Description

		np.ndarray
			Description

		np.ndarray
			Description

		Examples
		--------

		"""

		if self.R is None:
			raise ValueError('Cannot fit R: R=0.')

		X = []
		R_isotopes = pd.unique(self.R['isotope'])
		time = np.insert(np.unique(self.R['time']), 0, [0.0])

		for ip in R_isotopes:
			A0 = {p:0.0 for p in self.A0}
			for n,dt in enumerate(time[1:]-time[:-1]):
				_R_dict = {ip:self.R[self.R['isotope']==ip].iloc[n]['R']}
				A0 = {p:self.activity(p, dt, _R_dict=_R_dict, _A_dict=A0) for p in self.A0}

			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in self.counts.iterrows()])

		X = np.array(X)
		Y = self.counts['counts'].to_numpy()
		dY = self.counts['unc_counts'].to_numpy()

		func = lambda X_f, *R_f: np.dot(np.asarray(R_f), X_f)
		p0 = np.ones(len(X))
		fit, cov = curve_fit(func, X, Y, sigma=dY, p0=p0, bounds=(0.0*p0, np.inf*p0))

		for n,ip in enumerate(R_isotopes):
			df_sub = self.R[self.R['isotope']==ip]
			self.R.loc[df_sub.index, 'R'] = df_sub['R']*fit[n]

		for n,dt in enumerate(time[1:]-time[:-1]):
			_R_dict = {p:self.R[self.R['isotope']==p].iloc[n]['R'] for p in pd.unique(self.R['isotope'])}
			self.A0 = {p:self.activity(p, dt, _R_dict=_R_dict) for p in self.A0}

		R_avg = self.R_avg
		R_norm = np.array([R_avg[R_avg['isotope']==i]['R_avg'].to_numpy()[0] for i in R_isotopes])
		return R_isotopes, R_norm, cov*(R_norm/fit)**2
		
	def fit_A0(self):
		""" Description

		...

		Returns
		-------
		list
			Description

		np.ndarray
			Description

		np.ndarray
			Description

		Examples
		--------

		"""

		if self.R is not None:
			raise ValueError('Cannot fit A0 when R!=0.')

		X = []
		A0_isotopes = [i for i in self.A0]
		for ip in A0_isotopes:
			A0 = {p:(self.A0[p] if p==ip else 0.0) for p in self.A0}
			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in self.counts.iterrows()])

		X = np.array(X)
		Y = self.counts['counts'].to_numpy()
		dY = self.counts['unc_counts'].to_numpy()

		func = lambda X_f, *R_f: np.dot(np.asarray(R_f), X_f)
		p0 = np.ones(len(X))
		fit, cov = curve_fit(func, X, Y, sigma=dY, p0=p0, bounds=(0.0*p0, np.inf*p0))

		for n,ip in enumerate(A0_isotopes):
			self.A0[ip] *= fit[n]

		A_norm = np.array([self.A0[i] for i in A0_isotopes])
		return A0_isotopes, A_norm, cov*(A_norm/fit)**2
		
	def plot(self, time=None, max_plot=None, max_label=10, **kwargs):
		""" Description

		...

		Parameters
		----------
		time : array_like, optional
			Description of x

		max_plot : int, optional
			Description

		max_label : int, optional
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

		if time is not None:
			time = np.asarray(time)
		elif self.counts is None:
			time = np.linspace(0, 5.0*np.log(2)/self.chain[0,0], 1000)
		else:
			time = np.linspace(0, 1.25*self.counts['stop'].max(), 1000)

		if max_plot is None:
			max_plot = len(self.isotopes)

		ordr = int(np.floor(np.log10(np.average(self.activity(self.isotopes[0], time)))/3.0))
		lb_or = {-4:'p',-3:'n',-2:r'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G',4:'T'}[ordr]
		mult = 10**(-3*ordr)

		if self.R is not None:
			A0 = {p:0.0 for p in self.A0}
			T = np.insert(np.unique(self.R['time']), 0, [0.0])
			T_grid = np.array([])
			A_grid = {p:np.array([]) for p in A0}

			for n,dt in enumerate(T[1:]-T[:-1]):
				_R_dict = {p:self.R[self.R['isotope']==p].iloc[n]['R'] for p in pd.unique(self.R['isotope'])}
				dT = np.linspace(0, dt, 50)
				T_grid = dT if n==0 else np.append(T_grid, T_grid[-1]+dT)
				A_grid = {p:np.append(A_grid[p], self.activity(p, dT, _R_dict=_R_dict, _A_dict=A0)) for p in A0}
				A0 = {p:A_grid[p][-1] for p in A0}


		plot_time = time if self.R is None else np.append(T_grid-T_grid[-1], time.copy())
		for n,istp in enumerate(self.isotopes):
			if self.chain[n,0]>1E-12 and n<max_plot:
				A = self.activity(istp, time)
				if self.R is not None:
					A = np.append(A_grid[istp], A)

				label = Isotope(istp).TeX if n<max_label else None
				line, = ax.plot(plot_time, A*mult, label=label)

				if self.counts is not None:
					df = self.counts[self.counts['isotope']==istp]
					if len(df):
						x, y, yerr = df['start'].to_numpy(), df['activity'].to_numpy(), df['unc_activity'].to_numpy()
						idx = np.where((0.3*y>yerr)&(yerr>0.0)&(np.isfinite(yerr))&((self.activity(istp, x)-y)**2/yerr**2<10.0))
						x, y, yerr = x[idx], y[idx], yerr[idx]
					
						ax.errorbar(x, y*mult, yerr=yerr*mult, ls='None', marker='o', color=line.get_color(), label=None)



		ax.set_xlabel('Time ({})'.format(self.units))
		ax.set_ylabel('Activity ({}Bq)'.format(lb_or))
		ax.legend(loc=0)
		return _draw_plot(f, ax, **kwargs)
		