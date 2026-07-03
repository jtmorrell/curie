from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import datetime as dtm
pd.options.mode.chained_assignment = None

from scipy.optimize import curve_fit

from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from .isotope import Isotope
from .spectrum import Spectrum

class DecayChain(object):
	"""Radioactive Decay Chain

	Uses the Bateman equations to calculate the activities and number of decays
	from a radioactive decay chain as a function of time, both in production
	and decay.  Also, initial isotope activities and production rates can
	be fit to observed count data, or directly fit to HPGe spectra using the
	`get_counts()` function.
	
	Parameters
	----------
	parent_isotope : str
		Parent isotope in the chain.

	R : array_like, dict, str or pd.DataFrame
		Production rate for each isotope in the decay chain as a function of time.
		If a Nx2 np.ndarray, element n gives the production rate R_n up until
		time t_n for the parent isotope. E.g. If the production rate of the parent
		is 5 for 1 hour, and 8 for 3 hours, the array will be [[5, 1], [8, 4]].  If
		instead time intervals are preferred to a monotonically increasing grid
		of timestamps, set 'timestamp=False'.  In this case the production rate array
		will be [[5, 1], [8, 3]]. (R=5 for 1 hour, R=8 for 3 hours).

		If R is a dict, it specifies the production rate for multiple isotopes, 
		where the keys are the isotopes and the values are type np.ndarray.

		If R is a pd.DataFrame, it must have columns 'R' and 'time', and optionally 'isotope'
		if R>0 for any isotopes other than the parent.  If R is a str, it must be a 
		path to a file where the same data is provided.  Supported file types are
		.csv, .json and .db files, where .json files must be in the 'records' format,
		and .db files must have a table named 'R'.  Also, each isotope must have
		the same time grid, for both timestamp=True and timestamp=False.

	A0 : float or dict
		Initial activity.  If a float, the initial activity of the parent isotope.
		If a dict, the keys are the isotopes for which the values represent the
		initial activity.

	units : str, optional
		Units of time for the chain. Options are 'ns', 'us', 'ms', 's', 'm', 'h', 
		'd', 'y', 'ky', 'My', 'Gy'.  Default is 's'.

	timestamp : bool, optional
		Determines if the 'time' variable in R is to be read as a timestamp-like grid,
		i.e. in monotonically increasing order, or as a series of time intervals.
		Default is `True`.

	Attributes
	----------
	R : pd.DataFrame
		Production rate as a function of time, for each isotope in the chain. This
		will be modified if `fit_R()` is called.

	A0 : dict
		Initial activity of each isotope in the chain.

	isotopes : list
		List of isotopes in the decay chain.

	counts : pd.DataFrame
		Observed counts from isotopes in the decay chain, which can be used
		to determine the initial activities or average production rates using
		the `fit_R()` or `fit_A0()` functions.

	R_avg : pd.DataFrame
		Time-averaged production rate for each isotope where R>0.  This will be
		modified if `fit_R()` is called.

	Examples
	--------
	>>> dc = ci.DecayChain('Ra-225', R=[[1.0, 1.0], [0.5, 1.5], [2.0, 6]], units='d')
	>>> print(dc.isotopes)
	['225RAg', '225ACg', '221FRg', '217ATg', '213BIg', '217RNg', '209TLg', '213POg', '209PBg', '209BIg', '205TLg']
	>>> print(dc.R_avg)
          R_avg isotope
	0  1.708333  225RAg

	>>> dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
	>>> print(dc.isotopes)
	['152EUg', '152GDg', '152SMg', '148SMg', '144NDg', '140CEg']

	"""

	def __init__(self, parent_isotope, R=None, A0=None, units='s', timestamp=True):
		if units.lower() in ['hr','min','yr','sec']:
			units = {'hr':'h','min':'m','yr':'y','sec':'s'}[units.lower()]
		self.units = units
		

		istps = [Isotope(parent_isotope)]
		self.isotopes = [istps[0].name]
		self.R, self.A0, self._counts = None, {self.isotopes[0]:0.0}, None
		self._cal_data = {}
		self._chain = [[istps[0].decay_const(units), [], []]]
		stable_chain = [False]

		while not all(stable_chain):
			for n in [n for n,ch in enumerate(stable_chain) if not ch]:
				stable_chain[n] = True
				for prod in istps[n].decay_products:
					br = istps[n].decay_products[prod]
					I = Isotope(prod)
					if I.name in self.isotopes:
						self._chain[self.isotopes.index(I.name)][1].append(br)
						self._chain[self.isotopes.index(I.name)][2].append(n)
					else:
						istps.append(I)
						self.isotopes.append(I.name)
						self._chain.append([I.decay_const(units), [br], [n]])
						# chain expansion stops only at truly stable members, so chain
						# composition is independent of the chosen time units
						stable_chain.append(I.stable)
						if not stable_chain[-1]:
							self.A0[self.isotopes[-1]] = 0.0
		self._chain = np.array(self._chain, dtype=object)
		self._branches = self._generate_branches()

		if A0 is not None:
			if type(A0)==float or type(A0)==int:
				self.A0[self.isotopes[0]] = float(A0)
			else:
				for i in A0:
					self.A0[self._filter_name(i)] = float(A0[i])


		if R is not None:
			if type(R)==str:
				if R.endswith('.json'):
					self.R = pd.DataFrame(json.loads(open(R).read()))
				elif R.endswith('.csv'):
					self.R = pd.read_csv(R, header=0).ffill()
				elif R.endswith('.db'):
					self.R = pd.read_sql('SELECT * FROM R', _get_connection(R))

				if 'isotope' not in self.R.columns.to_list():
					self.R['isotope'] = self.isotopes[0]

			elif type(R)==pd.DataFrame:
				self.R = R.copy(deep=True)
				if 'isotope' not in self.R.columns.to_list():
					self.R['isotope'] = self.isotopes[0]

			elif type(R)==dict:
				self.R = pd.DataFrame({'isotope':[], 'R':[], 'time':[]})
				for ip in R:
					rate = np.array(R[ip])
					rt = pd.DataFrame({'isotope':self._filter_name(ip),'R':rate[:,0],'time':rate[:,1]})
					self.R = pd.concat([self.R, rt], ignore_index=True).reset_index(drop=True)

			elif type(R)==list or type(R)==np.ndarray:
				R = np.asarray(R)
				self.R = pd.DataFrame({'isotope':self.isotopes[0], 'R':R[:,0], 'time':R[:,1]})

			self.R['isotope'] = [self._filter_name(i) for i in self.R['isotope']]

			if not timestamp:
				for ip in pd.unique(self.R['isotope']):
					self.R.loc[self.R['isotope']==ip,'time'] = np.cumsum(self.R.loc[self.R['isotope']==ip,'time'])

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

	def _generate_branches(self):
		daughters = {i:[] for i in range(len(self._chain))}
		for i,pars in enumerate(self._chain[1:,2]):
			for p in pars:
				daughters[p].append(i+1)

		branches = [[0]]
		stable_chain = [len(daughters[br[-1]])==0 for br in branches]
		while not all(stable_chain):
			for par in list(set([b[-1] for b in branches])):
				ds = daughters[par]
				if len(ds)==0:
					continue
				if len(ds)>1:
					to_dup = [br for br in branches if br[-1]==par]
					for m in range(len(ds)-1):
						for b in to_dup:
							branches.append(b+[ds[m+1]])
					for br in branches:
						if br[-1]==par:
							br.append(ds[0])
				else:
					for br in branches:
						if br[-1]==par:
							br.append(ds[0])
			stable_chain = [len(daughters[br[-1]])==0 for br in branches]

		br_ratios = []
		for br in branches:
			r = []
			for n,i in enumerate(br[:-1]):
				r.append(self._chain[br[n+1]][1][self._chain[br[n+1]][2].index(i)])
			br_ratios.append(r+[0.0])

		return branches, br_ratios

	def _get_branches(self, istp):
		if self._filter_name(istp) not in self.isotopes:
			return [], []

		m = self._index(istp)
		branches, br_ratios = [], []

		for n,br in enumerate(self._branches[0]):
			if m in br:
				k = br.index(m)
				new_br = np.array(br[:k+1])
				if not any([np.array_equal(b, new_br) for b in branches]):
					branches.append(new_br)
					br_ratios.append(np.array(self._branches[1][n][:k] + [0.0]))
		return br_ratios, branches

	@staticmethod
	def _lm_clusters(lm):
		# Group indistinguishable decay constants (pairwise relative separation
		# below 1E-9): the standard partial-fraction Bateman coefficients are
		# singular for (near-)equal pairs, which occur in the database as
		# identically rounded half-lives. Relative comparison keeps the grouping
		# unit-independent.
		clusters = []
		for j in range(len(lm)):
			for cl in clusters:
				if abs(lm[j]-lm[cl[0]])<=1E-9*max(abs(lm[j]), abs(lm[cl[0]])):
					cl.append(j)
					break
			else:
				clusters.append([j])
		return clusters

	@staticmethod
	def _confluent_dd(m, lam, others, time, inv_lam):
		# (-1)^(m-1)/(m-1)! * (d/dlam)^(m-1) of exp(-lam*time)/(lam^inv_lam * P(lam)),
		# with P(lam) = prod(others-lam): the exact confluent Bateman coefficient for
		# a cluster of m equal decay constants (the divided difference of the
		# partial-fraction terms collapses to this derivative as the constants
		# coincide). Evaluated with the logarithmic-derivative recursion
		# f^(k) = sum_r C(k-1,r-1) * (ln f)^(r) * f^(k-r), which is free of the
		# catastrophic cancellation of the raw partial-fraction sum.
		P = np.prod(others-lam) if len(others) else 1.0
		f = np.exp(-lam*time)/((lam if inv_lam else 1.0)*P)
		if m==1:
			return f
		w = [None]*m
		for r in range(1, m):
			w[r] = math.factorial(r-1)*(np.sum(1.0/(others-lam)**r) if len(others) else 0.0)
			if inv_lam:
				w[r] += (-1.0)**r*math.factorial(r-1)/lam**r
		w[1] = w[1]-time
		g = [f]
		for k in range(1, m):
			s = 0.0
			for r in range(1, k+1):
				s = s+math.comb(k-1, r-1)*w[r]*g[k-r]
			g.append(s)
		return (-1.0)**(m-1)/math.factorial(m-1)*g[m-1]

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
		"""Activity of an isotope in the chain

		Computes the activity of a given isotope in the decay chain at a
		given time.  Units of activity are in Bq.  Units of time must be either
		the units for the DecayChain (default 's'), or specified by the `units`
		keyword.

		Parameters
		----------
		isotope : str
			Isotope for which the activity is calculated.

		time : array_like
			Time to calculate the activity.  Units of time must be the same
			as the decay chain, or be given by `units`. Note that if R!=0, time=0 is 
			defined as the end of production time.  Else, if A0!=0, time=0
			is defined as the time at which the specified activities equaled
			A0.  t<0 is not allowed. 

		units : str, optional
			Units of time, if different from the units of the decay chain.

		Returns
		-------
		activity : np.ndarray
			Activity of the given isotope in Bq.

		Examples
		--------
		>>> dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
		>>> print(dc.activity('152EU', time=0))
		3700.0
		>>> print(dc.activity('152EU', time=13.537, units='y'))
		1849.999906346199

		"""

		time = np.asarray(time)
		A = np.zeros(len(time)) if time.shape else np.array(0.0)

		finished = []
		for m,(BR, chain) in enumerate(zip(*self._get_branches(isotope))):
			lm = np.asarray(self._r_lm(units)*self._chain[chain, 0], dtype=np.float64)
			L = len(chain)
			# every Bateman term is invariant under (lm, t) -> (lm/sc, t*sc), so a
			# branch-wise geometric-mean rescale centers the partial-fraction
			# products near unity and keeps long chains inside float64 range in
			# any choice of time units
			sc = np.exp(np.mean(np.log(lm[lm>0]))) if np.any(lm>0) else 1.0
			lm_s, time_s = lm/sc, time*sc
			for i in range(L):
				sub = tuple(chain[i:])
				if sub in finished:
					continue
				finished.append(sub)
				# if i==L-1 and m>0: # only add A0 of end isotope once
					# continue

				ip = self.isotopes[chain[i]]
				A0 = self.A0.get(ip, 0.0) if _A_dict is None else _A_dict.get(ip, 0.0)
				if A0==0.0 and (_R_dict is None or lm[i]==0.0):
					continue
				A_i = lm_s[-1]*(A0/lm_s[i])

				B_i = np.prod(lm_s[i:-1]*BR[i:-1])

				lms = lm_s[i:]
				for cl in self._lm_clusters(lms):
					lam, mc = np.mean(lms[cl]), len(cl)
					others = np.array([lms[k] for k in range(len(lms)) if k not in cl])
					A += A_i*B_i*self._confluent_dd(mc, lam, others, time_s, False)
					if _R_dict is not None:
						if ip in _R_dict:
							if mc==1:
								# expm1 makes (1-exp(-lam*t))/lam exact for every lam*t,
								# with no threshold branch; lam==0 is the exact limit t
								C = np.prod(others-lam) if len(others) else 1.0
								if lam==0.0:
									A += _R_dict[ip]*lm_s[-1]*B_i*time_s/C
								else:
									A += _R_dict[ip]*lm_s[-1]*B_i*(-np.expm1(-lam*time_s))/(lam*C)
							else:
								A += _R_dict[ip]*lm_s[-1]*B_i*(self._confluent_dd(mc, lam, others, 0.0, True)-self._confluent_dd(mc, lam, others, time_s, True))
		return A
		
	def decays(self, isotope, t_start, t_stop, units=None, _A_dict=None):
		"""Number of decays in a given time interval

		Computes the number of decays from a given isotope in the
		decay chain in the time interal t_start to t_stop.  The 
		units of t_start and t_stop must be either the same units
		as the decay chain, or be specified by the `units` keyword.

		Parameters
		----------
		isotope : str
			Isotope for which the number of decays is calculated.

		t_start : array_like
			Time of the start of the interval.

		t_stop : array_like
			Time of the end of the interval.

		units : str, optional
			Units of time, if different from the units of the decay chain.

		Returns
		-------
		decays : np.ndarray
			Number of decays

		Examples
		--------
		>>> dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
		>>> print(dc.decays('152EU', t_start=1, t_stop=2))
		13319883.293399204
		>>> print(dc.decays('152EU', t_start=50, t_stop=50.1, units='y'))
		900151618.5228329

		"""

		t_start, t_stop = np.asarray(t_start), np.asarray(t_stop)
		D = np.zeros(len(t_start)) if t_start.shape else (np.zeros(len(t_stop)) if t_stop.shape else np.array(0.0))

		for m,(BR, chain) in enumerate(zip(*self._get_branches(isotope))):
			lm = np.asarray(self._r_lm(units)*self._chain[chain,0], dtype=np.float64)
			L = len(chain)
			# branch-wise rescale as in activity(); the decay integral carries one
			# net power of time, so each rescaled term is divided by sc
			sc = np.exp(np.mean(np.log(lm[lm>0]))) if np.any(lm>0) else 1.0
			lm_s, t1_s, t2_s = lm/sc, t_start*sc, t_stop*sc
			for i in range(L):
				if i==L-1 and m>0:
					continue

				ip = self.isotopes[chain[i]]
				A0 = self.A0.get(ip, 0.0) if _A_dict is None else _A_dict.get(ip, 0.0)
				if A0==0.0:
					continue
				A_i = lm_s[-1]*(A0/lm_s[i])
				B_i = np.prod(lm_s[i:-1]*BR[i:-1])

				lms = lm_s[i:]
				for cl in self._lm_clusters(lms):
					lam, mc = np.mean(lms[cl]), len(cl)
					others = np.array([lms[k] for k in range(len(lms)) if k not in cl])
					if mc==1:
						# expm1 makes (exp(-lam*t1)-exp(-lam*t2))/lam exact for every
						# lam*t - a threshold branch on lam alone misclassifies slow
						# members at long times, where lam*t is order unity even though
						# lam is tiny; lam==0 is the exact linear limit
						C = np.prod(others-lam) if len(others) else 1.0
						if lam==0.0:
							D += A_i*B_i*(t2_s-t1_s)/(C*sc)
						else:
							D += A_i*B_i*np.exp(-lam*t1_s)*(-np.expm1(-lam*(t2_s-t1_s)))/(lam*C*sc)
					else:
						D += A_i*B_i*(self._confluent_dd(mc, lam, others, t1_s, True)-self._confluent_dd(mc, lam, others, t2_s, True))/sc

		return D*self._r_lm((self.units if units is None else units), True)

	@property
	def counts(self):
		return self._counts

	@counts.setter
	def counts(self, N_c):
		if N_c is not None:
			if type(N_c)==pd.DataFrame:
				self._counts = N_c.copy(deep=True)
				self._counts['isotope'] = [self._filter_name(ip) for ip in self._counts['isotope']]

			elif type(N_c)!=dict:
				N_c = np.asarray(N_c)
				self._counts = pd.DataFrame({'isotope':self.isotopes[0],
											'start':N_c[:,0],
											'stop':N_c[:,1],
											'counts':N_c[:,2],
											'unc_counts':N_c[:,3]})

			else:
				self._counts = pd.DataFrame({'isotope':[],'start':[],'stop':[],'counts':[],'unc_counts':[]})
				for ip in N_c:
					ct = np.array(N_c[ip])
					if len(ct.shape)==1:
						ct = np.array([ct])
					ct = pd.DataFrame({'isotope':self._filter_name(ip),
										'start':ct[:,0],
										'stop':ct[:,1],
										'counts':ct[:,2],
										'unc_counts':ct[:,3]})
					self._counts = pd.concat([self._counts, ct], ignore_index=True).reset_index(drop=True)

			dcy = [float(self.decays(p['isotope'], p['start'], p['stop'])) for n,p in self._counts.iterrows()]
			for d,(n,p) in zip(dcy, self._counts.iterrows()):
				if d==0.0:
					raise ValueError('Cannot assign counts to {}: no decays in the given interval (stable isotope, or zero activity).'.format(p['isotope']))
			self._counts['activity'] = [p['counts']*self.activity(p['isotope'], p['start'])/d for d,(n,p) in zip(dcy, self._counts.iterrows())]
			self._counts['unc_activity'] = self._counts['unc_counts']*self.counts['activity']/self._counts['counts']
			
		
	def get_counts(self, spectra, EoB, peak_data=None):
		"""Retrieves the number of measured decays

		Takes the number of measured decays from one of the following: a list of spectra,
		a file with peak data, or a pandas DataFrame with peak data.

		Parameters
		----------
		spectra : list or str
			List of ci.Spectrum objects, or str of spectra filenames.  If list of str, 
			peak_data **must** be specified.  In this case the filenames must be
			an exact match of the filenames in `peak_data`.  If spectra is a str,
			it is assumed to be a regex match for the filenames in `peak_data`.
		
		EoB : str or datetime.datetime
			Date/time of end-of-bombardment (t=0).  Must be a datetime object or
			a string in the format '%m/%d/%Y %H:%M:%S'.  This is used to calculate
			the decay time for the count.

		peak_data : str or pd.DataFrame, optional
			Either a file path to a file that was created using
			`ci.Spectrum.saveas()` or a DataFrame with the same 
			structure as `ci.Spectrum.peaks`.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']
		>>> sp.saveas('test_spec.json')

		>>> dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
		>>> dc.get_counts([sp], EoB='01/01/2016 08:39:08')
		>>> dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')
		>>> print(dc.counts)

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


		if type(spectra)==str and peak_data is not None:
			df = peak_data['filename']
			spectra = list(set(map(str, df[df.str.contains(spectra)].to_list())))


		for sp in spectra:
			if type(sp)==str:
				if peak_data is not None:

					df = peak_data[peak_data['filename']==sp]
					df['isotope'] = [self._filter_name(i) for i in df['isotope']]
					df = df[df['isotope'].isin(self.isotopes)]

					if len(df):
						start_time = df.iloc[0]['start_time']
						if isinstance(start_time, str):
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
			if len(df):
				ct = pd.DataFrame({'isotope':df['isotope'], 'start':start, 'stop':stop, 'counts':df['decays'], 'unc_counts':df['unc_decays']})
				# decompose the uncertainty by correlation class where the peak data
				# carries the provenance: counting (independent), gamma intensity
				# (correlated within a line), efficiency (correlated within a
				# calibration) - used by the generalized-least-squares fits
				dec_cols = {'counts','unc_counts','intensity','unc_intensity','efficiency','unc_efficiency','energy'}
				if dec_cols.issubset(df.columns):
					ct['energy'] = df['energy'].to_numpy()
					ct['unc_stat'] = (df['decays']*df['unc_counts']/df['counts']).to_numpy()
					ct['unc_line'] = (df['decays']*df['unc_intensity']/df['intensity']).to_numpy()
					ct['line'] = [i+':'+'{:.2f}'.format(e) for i,e in zip(df['isotope'], df['energy'])]
					ct['unc_corr'] = (df['decays']*df['unc_efficiency']/df['efficiency']).to_numpy()
					if type(sp)!=str:
						cal = 'effcal:'+','.join('{:.9g}'.format(p) for p in sp.cb.effcal)
						self._cal_data[cal] = (np.asarray(sp.cb.effcal, dtype=np.float64), np.asarray(sp.cb.unc_effcal, dtype=np.float64))
						ct['cal'] = cal
					elif 'effcal' in df.columns:
						ct['cal'] = ('effcal:'+df['effcal'].astype(str)).to_numpy()
					else:
						ct['cal'] = ''
				counts.append(ct)

		self.counts = pd.concat(counts, sort=True, ignore_index=True).sort_values(by=['start']).reset_index(drop=True)


	@staticmethod
	def _eff_correlation(cal, energies):
		# correlation of efficiency errors between energies, from the efficiency
		# calibration's parameter covariance; falls back to full correlation when
		# the covariance is unusable
		from .calibration import Calibration
		effcal, unc_effcal = cal
		if unc_effcal.shape != (len(effcal), len(effcal)) or not np.all(np.isfinite(unc_effcal)):
			return np.ones((len(energies), len(energies)))
		cb = Calibration()
		eps = 1E-8
		f0 = cb.eff(energies, effcal)
		G = []
		for i in range(len(effcal)):
			p = np.array(effcal, dtype=np.float64)
			p[i] += eps
			G.append((cb.eff(energies, p)-f0)/eps)
		G = np.array(G)
		C = G.T @ unc_effcal @ G
		d = np.sqrt(np.clip(np.diag(C), 1E-300, None))
		return np.clip(C/np.outer(d, d), -1.0, 1.0)

	def _counts_covariance(self, fc, m, corr=None, corr_group=None, norm_frac=1.0):
		# Covariance of the count data for the generalized-least-squares fits:
		# diagonal counting variance plus correlated blocks for shared gamma
		# intensities (within a line; the isotope-common normalization fraction
		# norm_frac defaults to fully common) and shared efficiencies (within a
		# calibration, using its parameter covariance when captured by
		# get_counts). Magnitudes come from the fitted model values m rather than
		# the measured counts, which would bias correlated fits low (Peelle's
		# pertinent puzzle). Returns None if only total uncertainties are known.
		y = fc['counts'].to_numpy()
		dec = ['unc_stat','unc_line','line','unc_corr','cal']
		if all(c in fc.columns for c in dec) and fc[['unc_stat','unc_line','unc_corr']].notna().to_numpy().all():
			V = np.diag((m*(fc['unc_stat']/y).to_numpy())**2)
			r_line = m*(fc['unc_line']/y).to_numpy()
			for key, frac in [('line', 1.0-norm_frac), ('isotope', norm_frac)]:
				if frac<=0.0:
					continue
				for g in pd.unique(fc[key]):
					u = np.where((fc[key]==g).to_numpy(), r_line, 0.0)*np.sqrt(frac)
					V += np.outer(u, u)
			r_eff = m*(fc['unc_corr']/y).to_numpy()
			for g in pd.unique(fc['cal']):
				msk = (fc['cal']==g).to_numpy()
				if g in self._cal_data and 'energy' in fc.columns:
					ix = np.where(msk)[0]
					rho = self._eff_correlation(self._cal_data[g], fc['energy'].to_numpy()[ix])
					V[np.ix_(ix, ix)] += np.outer(r_eff[ix], r_eff[ix])*rho
				else:
					u = np.where(msk, r_eff, 0.0)
					V += np.outer(u, u)
		elif corr is not None:
			c = min(float(corr), 0.999999)
			u = m*(fc['unc_counts']/y).to_numpy()
			V = np.diag((1.0-c)*u**2)
			if corr_group is not None and corr_group in fc.columns:
				for g in pd.unique(fc[corr_group]):
					ug = np.where((fc[corr_group]==g).to_numpy(), u, 0.0)
					V += c*np.outer(ug, ug)
			else:
				V += c*np.outer(u, u)
		else:
			return None
		d = np.diag(V)
		V[np.diag_indices_from(V)] = d + 1E-12*np.max(d)
		return V

	def _gls_fit(self, X, Y, dY, fc, p0, corr, corr_group, norm_frac, scale_factor, cov):
		# Shared fitting core for fit_R/fit_A0: diagonal absolute-sigma pre-fit
		# (also the bare-counts path), then the generalized fit against the
		# assembled covariance, with a one-sided chi-square scale factor.
		func = lambda X_f, *R_f: np.dot(np.asarray(R_f), X_f)
		fit, cv = curve_fit(func, X, Y, sigma=dY, p0=p0, bounds=(0.0, np.inf), absolute_sigma=True)
		if cov is not None:
			V = np.asarray(cov, dtype=np.float64)
		else:
			V = self._counts_covariance(fc, np.abs(np.dot(fit, X)), corr, corr_group, norm_frac)
		if V is None:
			print('WARNING: counts carry only total uncertainties, so correlated systematics (shared gamma intensities, efficiencies) cannot be separated and unc_R may be underestimated. Provide decomposition columns (see get_counts) or the corr= option.')
			r = Y-np.dot(fit, X)
			chi2 = np.sum(r**2/dY**2)
		else:
			fit, cv = curve_fit(func, X, Y, sigma=V, p0=p0, bounds=(0.0, np.inf), absolute_sigma=True)
			r = Y-np.dot(fit, X)
			chi2 = float(r @ np.linalg.solve(V, r))
		dof = len(Y)-len(fit)
		if scale_factor and dof>0 and np.isfinite(chi2) and chi2/dof>1.0:
			cv = cv*(chi2/dof)
		return fit, cv

	@property	
	def R_avg(self):
		df = []
		for ip in np.unique(self.R['isotope']):
			time = np.insert(np.unique(self.R['time']), 0, [0.0])
			df.append({'isotope':ip, 'R_avg':np.average(self.R[self.R['isotope']==ip]['R'], weights=time[1:]-time[:-1])})
		return pd.DataFrame(df)
		
	def fit_R(self, max_error=0.4, min_counts=1, corr=None, corr_group=None, norm_frac=1.0, scale_factor=True, cov=None):
		"""Fit the production rate to count data

		Fits a scalar multiplier to the production rate (as a function of time) for
		each isotope specified in self.R.  The fit minimizes to the number of
		measured decays (self.counts) as a function of time, rather than the 
		activity, because the activity at each time point may be sensitive to
		the shape of the decay curve.

		Parameters
		----------
		max_error : float, optional
			The maximum relative error of a count datum to include in the fit. E.g. 0.25=25%
			(ony points with less than 25% error will be shown). Default, 0.4 (40%).

		min_counts : float or int, optional
			The minimum number of counts (decays) for a datum in self.counts to be included
			in the fit. Default, 1.

		corr : float, optional
			Opt-in uniform-correlation mode for counts that carry only total
			uncertainties: the fraction (0-1) of each point's variance treated as
			fully correlated across points (or within `corr_group` groups).
			Default `None` (off).

		corr_group : str, optional
			Name of a counts column defining the correlation groups for `corr`.
			Default `None` (global).

		norm_frac : float, optional
			Fraction of each line's intensity variance treated as common to all
			lines of the isotope (the decay-scheme normalization). Default 1.0
			(fully common - conservative until the intensity data carry the
			normalization uncertainty separately).

		scale_factor : bool, optional
			If `True` (default), the fitted covariance is inflated by chi-square
			per degree of freedom when that exceeds 1 (mutually inconsistent
			count data); it is never deflated.

		cov : array_like, optional
			Full covariance matrix of the count data, overriding the assembled
			one. For advanced use.


		Returns
		-------
		isotopes : list
			List of isotopes where R>0.  Same indices as fit. (i.e. isotope[0] corresponds
			to fit[0] and cov[0][0].)

		fit : np.ndarray
			The fitted time-averaged production rate for each isotope where R>0.

		cov : np.ndarray
			Covariance matrix on the fit.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
		>>> dc.get_counts([sp], EoB='01/01/2016 08:39:08')
		>>> print(dc.fit_R())
		(array(['152EUg'], dtype=object), array([1291584.51735774]), array([[1.67412376e+09]]))

		"""

		if self.R is None:
			raise ValueError('Cannot fit R: R=0.')

		X = []
		R_isotopes = [i for i in self.isotopes if i in pd.unique(self.R['isotope'])]
		time = np.insert(np.unique(self.R['time']), 0, [0.0])
		filter_counts = self.counts[(self.counts['counts']>min_counts)&(self.counts['unc_counts']<max_error*self.counts['counts'])]

		for ip in R_isotopes:
			A0 = {p:0.0 for p in self.A0}
			for n,dt in enumerate(time[1:]-time[:-1]):
				_R_dict = {ip:self.R[self.R['isotope']==ip].iloc[n]['R']}
				A0 = {p:self.activity(p, dt, _R_dict=_R_dict, _A_dict=A0) for p in self.A0}

			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in filter_counts.iterrows()])

		X = np.array(X)
		Y = filter_counts['counts'].to_numpy()
		dY = filter_counts['unc_counts'].to_numpy()

		
		wh = np.array([np.all(x>0) for x in X.T])
		p0 = np.average(Y[wh]/X[:,wh], axis=1)
		p0 = np.where((p0>0)&(np.isfinite(p0)), p0, 1.0)

		fit, cov = self._gls_fit(X, Y, dY, filter_counts, p0, corr, corr_group, norm_frac, scale_factor, cov)


		for n,ip in enumerate(R_isotopes):
			df_sub = self.R[self.R['isotope']==ip]
			self.R.loc[df_sub.index, 'R'] = df_sub['R']*fit[n]

		self.A0 = {i:0.0 for i in self.A0}
		for n,dt in enumerate(time[1:]-time[:-1]):
			_R_dict = {p:self.R[self.R['isotope']==p].iloc[n]['R'] for p in pd.unique(self.R['isotope'])}
			self.A0 = {p:self.activity(p, dt, _R_dict=_R_dict) for p in self.A0}

		self.counts = self.counts

		R_avg = self.R_avg
		R_norm = np.array([R_avg[R_avg['isotope']==i]['R_avg'].to_numpy()[0] for i in R_isotopes])
		if not np.any(np.isfinite(np.diag(cov))):
			cov = np.ones(cov.shape)*((np.average(dY/Y))*fit)**2
		return R_isotopes, R_norm, cov*(R_norm/fit)**2
		
	def fit_A0(self, max_error=0.4, min_counts=1, corr=None, corr_group=None, norm_frac=1.0, scale_factor=True, cov=None):
		"""Fit the initial activity to count data

		Fits a scalar multiplier to the initial activity for
		each isotope specified in self.A0.  The fit minimizes to the number of
		measured decays (self.counts) as a function of time, rather than the 
		activity, because the activity at each time point may be sensitive to
		the shape of the decay curve.

		Parameters
		----------
		max_error : float, optional
			The maximum relative error of a count datum to include in the fit. E.g. 0.25=25%
			(ony points with less than 25% error will be shown). Default, 0.4 (40%).

		min_counts : float or int, optional
			The minimum number of counts (decays) for a datum in self.counts to be included
			in the fit. Default, 1.

		corr : float, optional
			Opt-in uniform-correlation mode for counts that carry only total
			uncertainties: the fraction (0-1) of each point's variance treated as
			fully correlated across points (or within `corr_group` groups).
			Default `None` (off).

		corr_group : str, optional
			Name of a counts column defining the correlation groups for `corr`.
			Default `None` (global).

		norm_frac : float, optional
			Fraction of each line's intensity variance treated as common to all
			lines of the isotope (the decay-scheme normalization). Default 1.0
			(fully common - conservative until the intensity data carry the
			normalization uncertainty separately).

		scale_factor : bool, optional
			If `True` (default), the fitted covariance is inflated by chi-square
			per degree of freedom when that exceeds 1 (mutually inconsistent
			count data); it is never deflated.

		cov : array_like, optional
			Full covariance matrix of the count data, overriding the assembled
			one. For advanced use.


		Returns
		-------
		isotopes : list
			List of isotopes where A0>0.  Same indices as fit. (i.e. isotope[0] corresponds
			to fit[0] and cov[0][0].)

		fit : np.ndarray
			The initial activity for each isotope where A0>0.

		cov : np.ndarray
			Covariance matrix on the fit.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
		>>> dc.get_counts([sp], EoB='01/01/2016 08:39:08')
		>>> print(dc.fit_A0())
		(['152EUg'], array([6501.93665952]), array([[42425.53832341]]))

		"""

		if self.R is not None:
			raise ValueError('Cannot fit A0 when R!=0.')

		X = []
		A0_isotopes = [i for i in self.isotopes if i in self.A0]
		filter_counts = self.counts[(self.counts['counts']>min_counts)&(self.counts['unc_counts']<max_error*self.counts['counts'])]
		for ip in A0_isotopes:
			A0 = {p:(self.A0[p] if p==ip else 0.0) for p in self.A0}
			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in filter_counts.iterrows()])

		X = np.array(X)
		Y = filter_counts['counts'].to_numpy()
		dY = filter_counts['unc_counts'].to_numpy()

		p0 = np.ones(len(X))
		fit, cov = self._gls_fit(X, Y, dY, filter_counts, p0, corr, corr_group, norm_frac, scale_factor, cov)

		for n,ip in enumerate(A0_isotopes):
			self.A0[ip] *= fit[n]

		self.counts = self.counts

		A_norm = np.array([self.A0[i] for i in A0_isotopes])
		return A0_isotopes, A_norm, cov*(A_norm/fit)**2
		
	def plot(self, time=None, max_plot=10, max_label=10, max_plot_error=0.4, max_plot_chi2=10, **kwargs):
		"""Plot the activities in the decay chain

		Plots the activities as a function of time for all radioactive
		isotopes in the decay chain.  Can plot along a specified time
		grid, else the time will be inferred from the half-life of the
		parent isotope, or any count information given to self.counts.

		Parameters
		----------
		time : array_like, optional
			Time grid along which to plot.  Units must be the same as the decay chain.

		max_plot : int, optional
			Maximum number of isotope activities to plot in the decay chain. Default, 10.

		max_label : int, optional
			Maximum number of isotope activities to label in the legend. Default, 10.

		max_plot_error : float, optional
			The maximum relative error of a count point to include on the plot. E.g. 0.25=25%
			(ony points with less than 25% error will be shown). Default, 0.4.

		max_plot_chi2 : float or int, optional
			Maximum chi^2 of a count point to include on the plot. Only points with a chi^2
			less than this value will be shown. Default, 10.

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

		>>> dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
		>>> dc.get_counts([sp], EoB='01/01/2016 08:39:08')
		>>> dc.fit_R()
		>>> dc.plot()

		>>> dc = ci.DecayChain('99MO', A0=350E6, units='d')
		>>> dc.plot()

		"""

		f, ax = _init_plot(**kwargs)

		if time is not None:
			time = np.asarray(time)
		elif self.counts is None:
			time = np.linspace(0, 5.0*np.log(2)/self._chain[0,0], 1000)
		else:
			time = np.linspace(0, 1.25*self.counts['stop'].max(), 1000)

		if max_plot is None:
			max_plot = len(self.isotopes)

		ordr = int(np.floor(np.log10(np.average(self.activity(self.isotopes[0], time)))/3.0))
		lb_or = {-5:'f',-4:'p',-3:'n',-2:r'$\mu$',-1:'m',0:'',1:'k',2:'M',3:'G',4:'T',5:'E'}[ordr]
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
			if self._chain[n,0]>1E-12*self._r_lm(self.units, True) and n<max_plot:
				A = self.activity(istp, time)
				if self.R is not None:
					A = np.append(A_grid[istp], A)

				label = Isotope(istp).TeX if n<max_label else None
				line, = ax.plot(plot_time, A*mult, label=label)

				if self.counts is not None:
					df = self.counts[self.counts['isotope']==istp]
					if len(df):
						x, y, yerr = df['start'].to_numpy(), df['activity'].to_numpy(), df['unc_activity'].to_numpy()
						idx = np.where((max_plot_error*y>yerr)&(yerr>0.0)&(np.isfinite(yerr)))
						if len(x[idx]>0):
							x, y, yerr = x[idx], y[idx], yerr[idx]
						idx = np.where((self.activity(istp, x)-y)**2/yerr**2<max_plot_chi2)
						if len(x[idx]>0):
							x, y, yerr = x[idx], y[idx], yerr[idx]
					
						ax.errorbar(x, y*mult, yerr=yerr*mult, ls='None', marker='o', color=line.get_color(), label=None)



		ax.set_xlabel('Time ({})'.format(self.units))
		ax.set_ylabel('Activity ({}Bq)'.format(lb_or))
		ax.legend(loc=0)
		return _draw_plot(f, ax, **kwargs)
		