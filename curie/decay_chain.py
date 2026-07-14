
import json
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
from ._log import (_get_logger, _validate_config, NUMBER, NUMBER_OR_NONE, BOOLEAN,
				   STRING_OR_NONE, SEQUENCE_OR_NONE, MAPPING_OR_NONE)
from ._diagnostics import _diagnostics_frame

_log = _get_logger('decay_chain')

_FIT_CONFIG_SPEC = {'max_error': NUMBER, 'min_counts': NUMBER, 'corr': NUMBER_OR_NONE,
					'corr_group': STRING_OR_NONE, 'norm_frac': NUMBER,
					'scale_factor': BOOLEAN, 'max_chi2': NUMBER_OR_NONE,
					'exclude_lines': SEQUENCE_OR_NONE, 'time_range': MAPPING_OR_NONE,
					'unc_R_floor': NUMBER_OR_NONE}

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
		self._diagnostics = None
		self._fit_result = None
		self._fit_config = {'max_error':0.4, 'min_counts':1, 'corr':None,
							'corr_group':None, 'norm_frac':1.0, 'scale_factor':True,
							'max_chi2':None, 'exclude_lines':None, 'time_range':None,
							'unc_R_floor':None}
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
			
		
	def _loc(self, method):
		# message locator: names the chain by its parent isotope along with the
		# class and method, so interleaved output from loops over chains stays attributable
		return 'DecayChain({0}).{1}'.format(self.isotopes[0], method)

	@property
	def fit_config(self):
		return self._fit_config

	@fit_config.setter
	def fit_config(self, _fit_config):
		accepted = _validate_config(_fit_config, _FIT_CONFIG_SPEC, self._loc('fit_config'), _log)
		for nm in accepted:
			self._fit_config[nm] = accepted[nm]

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
			try:
				EoB = dtm.datetime.strptime(EoB, '%m/%d/%Y %H:%M:%S')
			except ValueError:
				raise ValueError(self._loc('get_counts')+": could not parse EoB {0!r} - expected '%m/%d/%Y %H:%M:%S' (e.g. 01/01/2016 08:39:08)".format(EoB))

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
			if not len(df):
				# routine when peak data spans many spectra and chains: detail at
				# DEBUG, matched-of-total in the summary line; all-empty raises below
				_log.debug(self._loc('get_counts')+': {0} contains no peaks matching chain isotopes [{1}]'.format(sp if type(sp)==str else sp.filename, ', '.join(self.isotopes)))
			if len(df):
				if start < 0:
					_log.warning(self._loc('get_counts')+': {0} starts {1:.4g} {2} before EoB - check EoB and the spectrum start time'.format(sp if type(sp)==str else sp.filename, -start, self.units))
				ct = pd.DataFrame({'isotope':df['isotope'], 'start':start, 'stop':stop, 'counts':df['decays'], 'unc_counts':df['unc_decays']})
				if 'chi2' in df.columns:
					# peak-fit quality rides with the count so the max_chi2
					# filter can act on it
					ct['chi2'] = df['chi2'].to_numpy()
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
						# the model rides in the identity string for non-vidmar
						# calibrations - a loglog parameter array can be the
						# same length as a vidmar one (byte-identical key for
						# the classic models)
						em = getattr(sp.cb, '_effcal_model', None)
						pfx = 'effcal:'+(em+':' if em is not None and em.startswith('loglog') else '')
						cal = pfx+','.join('{:.9g}'.format(p) for p in sp.cb.effcal)
						self._cal_data[cal] = (np.asarray(sp.cb.effcal, dtype=np.float64), np.asarray(sp.cb.unc_effcal, dtype=np.float64), em)
						ct['cal'] = cal
					elif 'effcal' in df.columns:
						# rows from files predating the effcal column group conservatively
						ct['cal'] = ('effcal:'+df['effcal'].fillna('').astype(str)).to_numpy()
					else:
						ct['cal'] = ''
				counts.append(ct)

		if not len(counts):
			raise ValueError(self._loc('get_counts')+': no counts found - none of the {0} spectra contain peaks matching [{1}].'.format(len(spectra), ', '.join(self.isotopes)))

		self.counts = pd.concat(counts, sort=True, ignore_index=True).sort_values(by=['start']).reset_index(drop=True)
		_log.info(self._loc('get_counts')+': {0} counts from {1} of {2} spectra for [{3}]'.format(len(self.counts), len(counts), len(spectra), ', '.join(self.isotopes)))


	@staticmethod
	def _eff_correlation(cal, energies):
		# correlation of efficiency errors between energies, from the efficiency
		# calibration's parameter covariance; falls back to full correlation when
		# the covariance is unusable
		from .calibration import Calibration
		effcal, unc_effcal, model = cal if len(cal)==3 else (cal[0], cal[1], None)
		if unc_effcal.shape != (len(effcal), len(effcal)) or not np.all(np.isfinite(unc_effcal)):
			return np.ones((len(energies), len(energies)))
		cb = Calibration()
		eps = 1E-8
		f0 = cb.eff(energies, effcal, model=model)
		G = []
		for i in range(len(effcal)):
			p = np.array(effcal, dtype=np.float64)
			p[i] += eps
			G.append((cb.eff(energies, p, model=model)-f0)/eps)
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
			Vi = np.diag((m*(fc['unc_stat']/y).to_numpy())**2)
			V = Vi.copy()
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
			Vi = np.diag((1.0-c)*u**2)
			V = Vi.copy()
			if corr_group is not None and corr_group in fc.columns:
				for g in pd.unique(fc[corr_group]):
					ug = np.where((fc[corr_group]==g).to_numpy(), u, 0.0)
					V += c*np.outer(ug, ug)
			else:
				V += c*np.outer(u, u)
		else:
			return None, None
		d = np.diag(V)
		V[np.diag_indices_from(V)] = d + 1E-12*np.max(d)
		return V, Vi

	def _gls_fit(self, X, Y, dY, fc, p0, corr, corr_group, norm_frac, scale_factor, cov, label='fit_R'):
		# Shared fitting core for fit_R/fit_A0: diagonal absolute-sigma pre-fit
		# (also the bare-counts path), then the generalized fit against the
		# assembled covariance, with a one-sided chi-square scale factor.
		# Returns (fit, covariance, chi2/dof of the unscaled fit, sigma scale
		# factor applied, scale note), where the note describes any applied
		# inflation for the summary line.
		func = lambda X_f, *R_f: np.dot(np.asarray(R_f), X_f)
		with np.errstate(all='ignore'):
			fit, cv = curve_fit(func, X, Y, sigma=dY, p0=p0, bounds=(0.0, np.inf), absolute_sigma=True)
			dof = len(Y)-len(fit)
			if cov is not None:
				V, Vi = np.asarray(cov, dtype=np.float64), None
			else:
				V, Vi = self._counts_covariance(fc, np.abs(np.dot(fit, X)), corr, corr_group, norm_frac)
			if V is None:
				_log.warning(self._loc(label)+': counts carry only total uncertainties, so correlated systematics (shared gamma intensities, efficiencies) cannot be separated and unc_R may be underestimated. Provide decomposition columns (see get_counts) or the corr= option.')
				r = Y-np.dot(fit, X)
				chi2n = np.sum(r**2/dY**2)/dof if dof>0 else np.inf
				scale, scale_note = 1.0, None
				if scale_factor and dof>0 and np.isfinite(chi2n) and chi2n>1.0:
					cv = cv*chi2n
					scale = float(np.sqrt(chi2n))
					scale_note = 'uncertainties scaled x{0:.2f}'.format(scale)
				return fit, cv, chi2n, scale, scale_note
			# One-sided scale factor: mutually inconsistent data can only indict the
			# INDEPENDENT error components (the correlated modes do not contribute to
			# point-to-point scatter), so only those are inflated, iterated until the
			# whitened chi-square per degree of freedom is consistent with 1.
			S2, chi2_fit = 1.0, np.inf
			for it in range(4):
				Vp = V if S2==1.0 else (V+(S2-1.0)*Vi if Vi is not None else V*S2)
				fit, cv = curve_fit(func, X, Y, sigma=Vp, p0=p0, bounds=(0.0, np.inf), absolute_sigma=True)
				r = Y-np.dot(fit, X)
				chi2n = float(r @ np.linalg.solve(Vp, r))/dof if dof>0 else np.inf
				if it==0:
					chi2_fit = chi2n
				if not (scale_factor and dof>0 and np.isfinite(chi2n) and chi2n>1.0):
					break
				S2 = S2*chi2n
		scale = float(np.sqrt(S2)) if S2!=1.0 else 1.0
		scale_note = 'independent uncertainty components scaled x{0:.2f}'.format(scale) if S2!=1.0 else None
		return fit, cv, chi2_fit, scale, scale_note

	def _filter_counts(self, cfg, label):
		# The count-selection filters shared by fit_R/fit_A0, each removal
		# announced at DEBUG and counted for the summary. The max_error/
		# min_counts selection is identical to
		# (counts>min_counts)&(unc_counts<max_error*counts).
		if self._counts is None:
			raise ValueError(self._loc(label)+' requires count data: set dc.counts or call dc.get_counts(...) first.')
		cts = self.counts
		max_error, min_counts = cfg['max_error'], cfg['min_counts']
		low = ~(cts['counts']>min_counts)
		err = (~low)&(~(cts['unc_counts']<max_error*cts['counts']))
		gone = low|err
		has_E = 'energy' in cts.columns

		def _line(c):
			if has_E and pd.notna(c['energy']):
				return ' ({0:.1f} keV)'.format(c['energy'])
			return ''

		for _, c in cts[low].iterrows():
			_log.debug(self._loc(label)+': dropped {0} count at t={1:.4g} {2}{3}: counts {4:.4g} <= min_counts {5}'.format(c['isotope'], c['start'], self.units, _line(c), c['counts'], min_counts))
		for _, c in cts[err].iterrows():
			_log.debug(self._loc(label)+': dropped {0} count at t={1:.4g} {2}{3}: relative error {4:.0f}% > max_error {5:.0f}%'.format(c['isotope'], c['start'], self.units, _line(c), 100.0*c['unc_counts']/c['counts'], 100.0*max_error))

		# named lines removed regardless of quality
		excl = pd.Series(False, index=cts.index)
		if cfg['exclude_lines'] is not None:
			for entry in cfg['exclude_lines']:
				if isinstance(entry, str):
					m = cts['isotope']==self._filter_name(entry)
					if not m.any():
						_log.warning(self._loc(label)+": exclude_lines entry '{0}' matches no counts - no {1} lines in the count data; entry ignored".format(entry, self._filter_name(entry)))
						continue
					excl |= m
				else:
					ip, E = self._filter_name(entry[0]), float(entry[1])
					if not has_E:
						_log.warning(self._loc(label)+': exclude_lines entry ({0}, {1:g}) needs the per-line energy column from decomposed count data (see get_counts); entry ignored'.format(ip, E))
						continue
					engs = cts.loc[cts['isotope']==ip, 'energy'].dropna().unique()
					if not len(engs):
						_log.warning(self._loc(label)+': exclude_lines entry ({0}, {1:g}) matches no counts - no {0} lines in the count data; entry ignored'.format(ip, E))
						continue
					near = float(engs[np.argmin(np.abs(engs-E))])
					if abs(near-E)>0.5:
						_log.warning(self._loc(label)+': exclude_lines entry ({0}, {1:g}) matches no line within 0.5 keV - nearest {0} line is {2:.1f} keV; entry ignored'.format(ip, E, near))
						continue
					excl |= (cts['isotope']==ip)&(cts['energy']==near)
		excl &= ~gone
		for _, c in cts[excl].iterrows():
			_log.debug(self._loc(label)+': excluded {0} count at t={1:.4g} {2}{3}: exclude_lines'.format(c['isotope'], c['start'], self.units, _line(c)))
		gone |= excl

		# counts outside a per-isotope time window (chain units, count start
		# time; None leaves that side open)
		tim = pd.Series(False, index=cts.index)
		if cfg['time_range'] is not None:
			for ip in cfg['time_range']:
				t_lo, t_hi = cfg['time_range'][ip]
				m = cts['isotope']==self._filter_name(ip)
				if t_lo is not None:
					tim |= m&(cts['start']<t_lo)
				if t_hi is not None:
					tim |= m&(cts['start']>t_hi)
		tim &= ~gone
		for _, c in cts[tim].iterrows():
			t_lo, t_hi = cfg['time_range'][[k for k in cfg['time_range'] if self._filter_name(k)==c['isotope']][0]]
			_log.debug(self._loc(label)+': excluded {0} count at t={1:.4g} {2}{3}: outside time_range ({4}, {5})'.format(c['isotope'], c['start'], self.units, _line(c), t_lo, t_hi))
		gone |= tim

		# counts whose originating peak fit was poor; counts without a chi2
		# column entry are unaffected (NaN fails the comparison)
		chi = pd.Series(False, index=cts.index)
		if cfg['max_chi2'] is not None:
			if 'chi2' not in cts.columns:
				_log.warning(self._loc(label)+': max_chi2 is set but the counts carry no peak-fit chi2 column (see get_counts) - filter has no effect')
			else:
				chi = (cts['chi2']>cfg['max_chi2'])&(~gone)
				for _, c in cts[chi].iterrows():
					_log.debug(self._loc(label)+': dropped {0} count at t={1:.4g} {2}{3}: peak fit chi2/dof {4:.3g} > max_chi2 {5:g}'.format(c['isotope'], c['start'], self.units, _line(c), c['chi2'], cfg['max_chi2']))
				gone |= chi

		drops = {'err':int(err.sum()), 'low':int(low.sum()), 'lines':int(excl.sum()),
				 'time':int(tim.sum()), 'chi2':int(chi.sum())}
		keep = cts[~gone]
		if not len(keep):
			raise ValueError(self._loc(label)+': 0 of {0} counts pass the filters ({1}). Loosen the filters (see dc.fit_config) or check the count data.'.format(len(cts), '; '.join(self._drop_parts(drops, cfg))))
		return keep, drops

	def _drop_parts(self, drops, cfg):
		# summary clauses for the count filters, in the order they are applied
		parts = []
		if drops['err']:
			parts.append('{0} dropped: relative error>{1:.0f}%'.format(drops['err'], 100.0*cfg['max_error']))
		if drops['low']:
			parts.append('{0} dropped: counts<={1}'.format(drops['low'], cfg['min_counts']))
		if drops['lines']:
			parts.append('{0} excluded: exclude_lines'.format(drops['lines']))
		if drops['time']:
			parts.append('{0} outside time_range'.format(drops['time']))
		if drops['chi2']:
			parts.append('{0} dropped: peak chi2>{1:g}'.format(drops['chi2'], cfg['max_chi2']))
		return parts

	def _log_gls_summary(self, label, what, n_fit, n_used, n_tot, drops, cfg, chi2, scale_note=None):
		# logs the fit summary and returns its body for the diagnostics table
		parts = self._drop_parts(drops, cfg)
		body = 'fit {0} {1} to {2}/{3} counts'.format(n_fit, what, n_used, n_tot)
		if parts:
			body += ' ({0})'.format('; '.join(parts))
		body += '; chi2/dof={0:.2g}'.format(chi2)
		if scale_note:
			body += '; '+scale_note
		_log.info(self._loc(label)+': '+body)
		return body

	def _diagnose_gls(self, label, what, isotopes, fit, p0, chi2, dof, n_points, n_dropped, scale, singular, body):
		# Per-isotope diagnostics rows for fit_R/fit_A0 (joint scalars repeated
		# on every row), surfacing the unmoved flag as a warning; returns the
		# assembled frame. Emitted-warning text rides in the message column.
		unmoved = np.isclose(fit, p0, rtol=1E-9, atol=0.0)
		par = 'R' if label=='fit_R' else 'A0'
		rows = []
		for n, ip in enumerate(isotopes):
			flags, msgs = [], [body]
			if fit[n]==0.0:
				flags.append('at_bound:'+par)
			if unmoved[n]:
				msg = '{0} {1} unchanged from initial estimate - fit may not have converged (flag: unmoved)'.format(ip, what)
				_log.warning(self._loc(label)+': '+msg)
				flags.append('unmoved')
				msgs.append(msg)
			if singular:
				flags.append('singular_cov')
				msgs.append('covariance estimate is singular - quoted uncertainties are unreliable (flag: singular_cov)')
			rows.append({'fit':label, 'chi2':(chi2 if (dof>0 and np.isfinite(chi2)) else np.nan), 'dof':int(dof),
						 'n_points':int(n_points), 'n_dropped':int(n_dropped), 'converged':True,
						 'model':'', 'scale_factor':float(scale), 'flags':','.join(flags),
						 'message':'; '.join(msgs), 'isotope':ip})
		return _diagnostics_frame(rows, extras={'isotope': object})

	@property
	def diagnostics(self):
		"""Fit diagnostics from the most recent fit_R or fit_A0 call

		Read-only pd.DataFrame with one row per fitted isotope (`fit` is
		'fit_R' or 'fit_A0'; the fit is joint, so chi2, dof, n_points,
		n_dropped and scale_factor repeat on every row): columns chi2
		(reduced, of the unscaled fit), dof, n_points (counts the fit used),
		n_dropped (counts removed by the max_error/min_counts filters),
		converged, model ('' - the chain fit has no model selection),
		scale_factor (uncertainty inflation applied;
		1.0 = none), flags (comma-joined, e.g. 'unmoved', 'at_bound:R',
		'singular_cov'), message (summary and warning text) and isotope.
		Empty (with the full schema) before any fit; rebuilt on each
		`fit_R()`/`fit_A0()` call.  Accessing it never triggers a fit.
		"""
		if self._diagnostics is None:
			return _diagnostics_frame(extras={'isotope': object})
		return self._diagnostics.copy()

	def _band_sigma(self, istp, time):
		# 1-sigma activity uncertainty from the stored fit covariance: the
		# activity is linear in the fitted multipliers, so the band is exact.
		# Basis curve b_i(t) = activity of istp produced by fitted isotope i
		# alone at the current (fitted) R/A0; var(t) = b C_rel b^T with C_rel
		# the relative covariance of the fitted multipliers. Returns None when
		# no usable fit record exists.
		fr = self._fit_result
		if fr is None:
			return None
		fit = np.asarray(fr['fit'], dtype=np.float64)
		cov = np.asarray(fr['cov'], dtype=np.float64)
		if not (np.all(fit>0) and np.all(np.isfinite(cov))):
			return None
		C_rel = cov/np.outer(fit, fit)
		B = []
		if fr['label']=='fit_R':
			time_R = np.insert(np.unique(self.R['time']), 0, [0.0])
			for ip in fr['isotopes']:
				A0 = {p:0.0 for p in self.A0}
				for n,dt in enumerate(time_R[1:]-time_R[:-1]):
					_R_dict = {ip:self.R[self.R['isotope']==ip].iloc[n]['R']}
					A0 = {p:self.activity(p, dt, _R_dict=_R_dict, _A_dict=A0) for p in self.A0}
				B.append(self.activity(istp, time, _A_dict=A0))
		else:
			for ip in fr['isotopes']:
				A0 = {p:(self.A0[p] if p==ip else 0.0) for p in self.A0}
				B.append(self.activity(istp, time, _A_dict=A0))
		B = np.asarray(B, dtype=np.float64)
		var = np.einsum('it,ij,jt->t', B, C_rel, B)
		return np.sqrt(var)

	@property
	def R_avg(self):
		df = []
		for ip in np.unique(self.R['isotope']):
			time = np.insert(np.unique(self.R['time']), 0, [0.0])
			df.append({'isotope':ip, 'R_avg':np.average(self.R[self.R['isotope']==ip]['R'], weights=time[1:]-time[:-1])})
		return pd.DataFrame(df)
		
	def fit_R(self, **kwargs):
		"""Fit the production rate to count data

		Fits a scalar multiplier to the production rate (as a function of time) for
		each isotope specified in self.R.  The fit minimizes to the number of
		measured decays (self.counts) as a function of time, rather than the
		activity, because the activity at each time point may be sensitive to
		the shape of the decay curve.

		Keyword arguments other than `p0` and `cov` are `fit_config` keys: they
		merge into `dc.fit_config` (like `Spectrum.fit_config`) and persist for
		subsequent fits.

		Parameters
		----------
		max_error : float, optional
			The maximum relative error of a count datum to include in the fit. E.g. 0.25=25%
			(ony points with less than 25% error will be shown). Default, 0.4 (40%).

		min_counts : float or int, optional
			The minimum number of counts (decays) for a datum in self.counts to be included
			in the fit. Default, 1.

		max_chi2 : float, optional
			Exclude counts whose originating peak fit had chi2/dof above this
			bound (needs the `chi2` column `get_counts` carries over from the
			peak data; counts without one are unaffected). Default `None` (off).

		exclude_lines : list, optional
			Named lines to exclude from the fit: entries are an isotope name
			('152EU') or an (isotope, energy_keV) tuple.  The energy matches the
			closest of that isotope's line energies in the counts, accepted
			within +-0.5 keV; an entry matching nothing excludes nothing and
			warns, naming the nearest candidate.  Default `None` (off).

		time_range : dict, optional
			Per-isotope count-time window, `{isotope: (t_lo, t_hi)}` in chain
			units; `None` for an open bound (e.g. `{'196AUm2': (None, 60.0)}`).
			Counts starting outside the window are excluded.  Default `None` (off).

		unc_R_floor : float, optional
			Relative floor on the returned production-rate uncertainty:
			unc_R >= floor x R, per isotope, applied after the fit (announced).
			E.g. 0.05 keeps every unc_R at or above 5%.  Default `None` (off).

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

		p0 : dict, optional
			Starting estimate override, `{isotope: R_estimate}` in production-rate
			units: seeds the fit at the given rate instead of the data-derived
			estimate (kwarg only - not a fit_config key). Default `None`.

		cov : array_like, optional
			Full covariance matrix of the count data, overriding the assembled
			one. For advanced use (kwarg only - not a fit_config key).


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
		(['152EUg'], array([27527059.31414273]), array([[2.03699956e+11]]))

		"""

		p0_user = kwargs.pop('p0', None)
		cov = kwargs.pop('cov', None)
		self.fit_config = kwargs
		cfg = self.fit_config

		if self.R is None:
			raise ValueError('Cannot fit R: R=0.')

		X = []
		R_isotopes = [i for i in self.isotopes if i in pd.unique(self.R['isotope'])]
		time = np.insert(np.unique(self.R['time']), 0, [0.0])
		filter_counts, drops = self._filter_counts(cfg, 'fit_R')

		for ip in R_isotopes:
			A0 = {p:0.0 for p in self.A0}
			for n,dt in enumerate(time[1:]-time[:-1]):
				_R_dict = {ip:self.R[self.R['isotope']==ip].iloc[n]['R']}
				A0 = {p:self.activity(p, dt, _R_dict=_R_dict, _A_dict=A0) for p in self.A0}

			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in filter_counts.iterrows()])

		X = np.array(X)
		Y = filter_counts['counts'].to_numpy()
		dY = filter_counts['unc_counts'].to_numpy()

		with np.errstate(divide='ignore', invalid='ignore'):
			wh = np.array([np.all(x>0) for x in X.T])
			p0 = np.average(Y[wh]/X[:,wh], axis=1)
			p0 = np.where((p0>0)&(np.isfinite(p0)), p0, 1.0)

		if p0_user is not None:
			# user seed is a production rate; the fitted parameter is a scalar
			# multiplier on the current R schedule, so convert through R_avg
			if not isinstance(p0_user, dict):
				raise ValueError(self._loc('fit_R')+': p0 must be a dict of {{isotope: R_estimate}} (got {0} {1!r})'.format(type(p0_user).__name__, p0_user))
			R_avg = self.R_avg
			for ip in p0_user:
				name = self._filter_name(ip)
				if name not in R_isotopes:
					_log.warning(self._loc('fit_R')+": p0 entry '{0}' is not a fitted isotope ([{1}]); entry ignored".format(ip, ', '.join(R_isotopes)))
					continue
				ra = float(R_avg[R_avg['isotope']==name]['R_avg'].to_numpy()[0])
				if ra>0 and np.isfinite(ra):
					p0[R_isotopes.index(name)] = float(p0_user[ip])/ra
					_log.debug(self._loc('fit_R')+': starting estimate for {0} seeded at R={1:.4g}'.format(name, float(p0_user[ip])))

		fit, cov, chi2, scale, scale_note = self._gls_fit(X, Y, dY, filter_counts, p0, cfg['corr'], cfg['corr_group'], cfg['norm_frac'], cfg['scale_factor'], cov, 'fit_R')
		body = self._log_gls_summary('fit_R', 'production rates', len(R_isotopes), len(filter_counts), len(self.counts), drops, cfg, chi2, scale_note)


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
		singular = not np.any(np.isfinite(np.diag(cov)))
		if singular:
			_log.warning(self._loc('fit_R')+': covariance estimate is singular - quoted uncertainties are unreliable (flag: singular_cov)')
			cov = np.ones(cov.shape)*((np.average(dY/Y))*fit)**2
		if cfg['unc_R_floor'] is not None:
			# relative floor on the multiplier-space diagonal: cov_norm and the
			# plot band inherit it (raising a diagonal keeps the matrix PSD)
			for n, ip in enumerate(R_isotopes):
				rel = np.sqrt(cov[n][n])/fit[n] if fit[n]>0 else np.inf
				if rel < cfg['unc_R_floor']:
					_log.info(self._loc('fit_R')+': unc_R for {0} raised to the floor {1:.3g}% of R (fit gave {2:.3g}%)'.format(ip, 100.0*cfg['unc_R_floor'], 100.0*rel))
					cov[n][n] = (cfg['unc_R_floor']*fit[n])**2
		dof = len(Y)-len(fit)
		self._diagnostics = self._diagnose_gls('fit_R', 'production rate', R_isotopes, fit, p0, chi2, dof,
											   len(filter_counts), sum(drops.values()), scale, singular, body)
		cov_norm = cov*(R_norm/fit)**2
		# private fit record: enough to reconstruct the fitted curve and its
		# uncertainty band (plotting), and which counts the fit used, without
		# re-running the fit
		self._fit_result = {'label':'fit_R', 'isotopes':R_isotopes, 'fit':fit, 'cov':cov, 'p0':p0,
							'chi2':chi2, 'dof':dof, 'scale_factor':scale, 'value':R_norm, 'cov_norm':cov_norm,
							'used':filter_counts.index.to_numpy()}
		return R_isotopes, R_norm, cov_norm
		
	def fit_A0(self, **kwargs):
		"""Fit the initial activity to count data

		Fits a scalar multiplier to the initial activity for
		each isotope specified in self.A0.  The fit minimizes to the number of
		measured decays (self.counts) as a function of time, rather than the
		activity, because the activity at each time point may be sensitive to
		the shape of the decay curve.

		Keyword arguments other than `cov` are `fit_config` keys: they merge
		into `dc.fit_config` (like `Spectrum.fit_config`) and persist for
		subsequent fits.

		Parameters
		----------
		max_error : float, optional
			The maximum relative error of a count datum to include in the fit. E.g. 0.25=25%
			(ony points with less than 25% error will be shown). Default, 0.4 (40%).

		min_counts : float or int, optional
			The minimum number of counts (decays) for a datum in self.counts to be included
			in the fit. Default, 1.

		max_chi2 : float, optional
			Exclude counts whose originating peak fit had chi2/dof above this
			bound (needs the `chi2` column `get_counts` carries over from the
			peak data; counts without one are unaffected). Default `None` (off).

		exclude_lines : list, optional
			Named lines to exclude from the fit: entries are an isotope name
			('152EU') or an (isotope, energy_keV) tuple.  The energy matches the
			closest of that isotope's line energies in the counts, accepted
			within +-0.5 keV; an entry matching nothing excludes nothing and
			warns, naming the nearest candidate.  Default `None` (off).

		time_range : dict, optional
			Per-isotope count-time window, `{isotope: (t_lo, t_hi)}` in chain
			units; `None` for an open bound.  Counts starting outside the
			window are excluded.  Default `None` (off).

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
			one. For advanced use (kwarg only - not a fit_config key).


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
		>>> itp, A0, cov = dc.fit_A0()
		>>> print(itp[0], A0[0])
		152EUg 138582.75831764835

		"""

		cov = kwargs.pop('cov', None)
		self.fit_config = kwargs
		cfg = self.fit_config

		if self.R is not None:
			raise ValueError('Cannot fit A0 when R!=0.')

		X = []
		A0_isotopes = [i for i in self.isotopes if i in self.A0]
		filter_counts, drops = self._filter_counts(cfg, 'fit_A0')
		for ip in A0_isotopes:
			A0 = {p:(self.A0[p] if p==ip else 0.0) for p in self.A0}
			X.append([self.decays(c['isotope'], c['start'], c['stop'], _A_dict=A0) for n,c in filter_counts.iterrows()])

		X = np.array(X)
		Y = filter_counts['counts'].to_numpy()
		dY = filter_counts['unc_counts'].to_numpy()

		p0 = np.ones(len(X))
		fit, cov, chi2, scale, scale_note = self._gls_fit(X, Y, dY, filter_counts, p0, cfg['corr'], cfg['corr_group'], cfg['norm_frac'], cfg['scale_factor'], cov, 'fit_A0')
		body = self._log_gls_summary('fit_A0', 'initial activities', len(A0_isotopes), len(filter_counts), len(self.counts), drops, cfg, chi2, scale_note)

		for n,ip in enumerate(A0_isotopes):
			self.A0[ip] *= fit[n]

		self.counts = self.counts

		A_norm = np.array([self.A0[i] for i in A0_isotopes])
		singular = not np.any(np.isfinite(np.diag(cov)))
		if singular:
			_log.warning(self._loc('fit_A0')+': covariance estimate is singular - quoted uncertainties are unreliable (flag: singular_cov)')
			cov = np.ones(cov.shape)*((np.average(dY/Y))*fit)**2
		dof = len(Y)-len(fit)
		self._diagnostics = self._diagnose_gls('fit_A0', 'initial activity', A0_isotopes, fit, p0, chi2, dof,
											   len(filter_counts), sum(drops.values()), scale, singular, body)
		cov_norm = cov*(A_norm/fit)**2
		self._fit_result = {'label':'fit_A0', 'isotopes':A0_isotopes, 'fit':fit, 'cov':cov, 'p0':p0,
							'chi2':chi2, 'dof':dof, 'scale_factor':scale, 'value':A_norm, 'cov_norm':cov_norm,
							'used':filter_counts.index.to_numpy()}
		return A0_isotopes, A_norm, cov_norm
		
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
			The maximum relative error of a count point to draw as a filled
			marker. E.g. 0.25=25%.  Points above the threshold are drawn as
			open grey markers rather than hidden.  Default, 0.4.

		max_plot_chi2 : float or int, optional
			Maximum chi^2 of a count point to draw as a filled marker.  Points
			above the threshold are drawn as open grey markers rather than
			hidden.  Default, 10.

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
		band_label, excl_label = r'fit $\pm 1\sigma$', 'excluded from fit/plot'
		excluded = []
		for n,istp in enumerate(self.isotopes):
			if self._chain[n,0]>1E-12*self._r_lm(self.units, True) and n<max_plot:
				A = self.activity(istp, time)
				if self.R is not None:
					A = np.append(A_grid[istp], A)

				label = Isotope(istp).TeX if n<max_label else None
				line, = ax.plot(plot_time, A*mult, label=label)

				sig = self._band_sigma(istp, time)
				if sig is not None:
					A_dec = self.activity(istp, time)
					ax.fill_between(time, (A_dec-sig)*mult, (A_dec+sig)*mult,
									color=line.get_color(), alpha=0.2, lw=0, label=band_label)
					band_label = None

				if self.counts is not None:
					df = self.counts[self.counts['isotope']==istp]
					if len(df):
						x, y, yerr = df['start'].to_numpy(), df['activity'].to_numpy(), df['unc_activity'].to_numpy()
						# points the fit or the plot thresholds set aside stay
						# visible as open grey markers - a clean-looking plot
						# must not hide points the fit had to contend with
						shown = np.ones(len(df), dtype=bool)
						if self._fit_result is not None and 'used' in self._fit_result:
							shown &= df.index.isin(self._fit_result['used'])
						with np.errstate(divide='ignore', invalid='ignore'):
							shown &= (max_plot_error*y>yerr)&(yerr>0.0)&(np.isfinite(yerr))
							shown &= (self.activity(istp, x)-y)**2/yerr**2<max_plot_chi2

						if shown.any():
							ax.errorbar(x[shown], y[shown]*mult, yerr=yerr[shown]*mult, ls='None', marker='o', color=line.get_color(), label=None)
						if (~shown).any():
							excluded.append((x[~shown], y[~shown]*mult, yerr[~shown]*mult))

		if excluded:
			# drawn after the y-limits are frozen from the fitted curves and
			# used points, and underneath them: a wild excluded point must not
			# set the scale or cover the data it was excluded from
			yl = ax.get_ylim()
			for x, y, yerr in excluded:
				ax.errorbar(x, y, yerr=yerr, ls='None', marker='o', mfc='none',
							color='0.6', alpha=0.7, zorder=1.5, label=excl_label)
				excl_label = None
			ax.set_ylim(yl)

		ax.set_xlabel('Time ({})'.format(self.units))
		ax.set_ylabel('Activity ({}Bq)'.format(lb_or))
		ax.legend(loc=0)
		return _draw_plot(f, ax, **kwargs)
		