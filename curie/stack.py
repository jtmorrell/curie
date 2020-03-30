from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import re
import numpy as np
import pandas as pd

from .data import _get_connection
from .plotting import _init_plot, _draw_plot
from .compound import Compound
from .element import Element

class Stack(object):
	"""Foil pack for stacked target calculations

	Computes the energy loss and (relative) charged particle flux through a stack
	of foils using the Anderson-Ziegler formulation for stopping powers.
	
	Parameters
	----------
	stack : list of dicts, pd.DataFrame or str
		Definition of the foils in the stack.  The 'compound' for each foil in
		the stack must be given, and the 'areal_density' or some combination of parameters
		that allow the areal density to be calculated must also be given.  Foils must
		also be given a 'name' if they are to be filtered by the .saveas(), .summarize(),
		and .plot() methods.  By default, foils without 'name' are not included by these
		methods.

		There are three acceptable formats for `stack`.  The first is a pd.DataFrame
		with the columns described. The second is a list of dicts, where each dict contains
		the appropriate keys.  The last is a str, which is a path to a file in either .csv,
		.json or .db format, where the headers of the file contain the correct information.
		Note that the .json file must follow the 'records' format (see pandas docs).  If a .db
		file, it must have a table named 'stack'.

		The 'areal_density' can be given directly, in units of mg/cm^2, or will be calculated
		from the following: 'mass' (in g) and 'area' (in cm^2), 'thickness' (in mm) and 'density'
		(in g/cm^3), or just 'thickness' if the compound is a natural element, or 
		is in `ci.COMPOUND_LIST` or the 'compounds' argument.

		Also, the following shorthand indices are supported: 'cm' for 'compound', 'd' for
		'density', 't' for 'thickness', 'm' for 'mass', 'a' for 'area', 'ad' for 'areal_density',
		and 'nm' for 'name'.


	particle : str
		Incident ion.  For light ions, options are 'p' (default), 'd', 't', 'a' for proton, 
		deuteron, triton and alpha, respectively.  Additionally, heavy ions can be
		specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'.For 
		light ions, the charge state is assumed to be fully stripped. For heavy ions
		the charge state is handled by a Bohr/Northcliffe parameterization consistent
		with the Anderson-Ziegler formalism.

	E0 : float
		Incident particle energy, in MeV.  If dE0 is not provided, it will
		default to 1 percent of E0.


	Other Parameters
	----------------
	compounds : str, pandas.DataFrame, list or dict
		Compound definitions for the compounds included in the foil stack.  If the compounds
		are not natural elements, or `ci.COMPOUND_LIST`, or if different weights or densities
		are required, they can be specified here. (Note specifying specific densities in the
		'stack' argument is probably more appropriate.)  Also, if the 'compound' name in the
		stack is a chemical formula, e.g. 'H2O', 'SrCO3', the weights can be inferred and 
		'compounds' doesn't need to be given.

		If compounds is a pandas DataFrame, it must have the columns 'compound', 'element', one of 
		'weight', 'atom_weight', or 'mass_weight', and optionally 'density'.  If a str, it must be
		a path to a .csv, .json or .db file, where .json files must be in the 'records' format and
		.db files must have a 'compounds' table.  All must have the above information.  For .csv 
		files, the compound only needs to be given for the first line of that compound definition.

		If compounds is a list, it must be a list of ci.Element or ci.Compound classes.  If it is a
		dict, it must have the compound names as keys, and weights as values, e.g. 
		{'Water':{'H':2, 'O':1}, 'Brass':{'Cu':-66,'Zn':-33}}

	dE0 : float
		1-sigma width of the energy distribution from which the initial
		particle energies are sampled, in MeV.  Default is to 1 percent of E0.

	N : int
		Number of particles to simulate. Default is 10000.

	dp : float
		Density multiplier.  dp is uniformly multiplied to all areal densities in the stack.  Default 1.0.

	chunk_size : int
		If N is large, split the stack calculation in to multiple "chunks" of size `chunk_size`. Default 1E7.

	accuracy : float
		Maximum allowed (absolute) error in the predictor-corrector method. Default 0.01.  If error is
		above `accuracy`, each foil in the stack will be solved with multiple steps, between `min_steps`
		and `max_steps`.

	min_steps : int
		The minimum number of steps per foil, in the predictor-corrector solver.  Default 2.

	max_steps : int
		The maximum number of steps per foil, in the predictor-corrector solver.  Default 50.

	Attributes
	----------
	stack : pandas.DataFrame
		'name', 'compound', 'areal_density', mean energy 'mu_E', and 1-sigma energy width 'sig_E'
		for each foil in the stack (energies in MeV).

	fluxes : pandas.DataFrame
		'flux' as a function of 'energy' for each foil in the stack where 'name' is not None.

	compounds : dict
		Dictionary with compound names as keys, and ci.Compound classes as values.

	Examples
	--------
	>>> stack = [{'cm':'H2O', 'ad':800.0, 'name':'water'},
				{'cm':'RbCl', 'density':3.0, 't':0.03, 'name':'salt'},
				{'cm':'Kapton', 't':0.025},
				{'cm':'Brass', 'm':3.5, 'a':1.0, 'name':'metal'}]
	>>> st = ci.Stack(stack, compounds='example_compounds.json')
	>>> st = ci.Stack(stack, compounds={'Brass':{'Cu':-66, 'Zn':-33}}, E0=60.0)
	>>> print(st.stack)
	    name compound  areal_density       mu_E     sig_E
	0  water      H2O         800.00  55.444815  2.935233
	1   salt     RbCl           9.00  50.668313  0.683532
	2    NaN   Kapton           3.55  50.612543  0.683325
	3  metal    Brass         350.00  49.159245  1.205481
	>>> st.saveas('stack_calc.csv')

	"""

	def __init__(self, stack, particle='p', E0=60.0, **kwargs):
		self._E0, self._particle = float(E0), particle
		self.compounds = {}
		self._parse_kwargs(**kwargs)

		if type(stack)==str:
			if stack.endswith('.json'):
				df = pd.read_json(stack, orient='records')

			elif stack.endswith('.csv'):
				df = pd.read_csv(stack, header=0)

			elif stack.endswith('.db'):
				df = pd.read_sql('SELECT * FROM stack', _get_connection(stack))

		elif type(stack)==list:
			df = pd.DataFrame(stack)

		elif type(stack)==pd.DataFrame:
			df = stack

		df = self._filter_cols(df)
		for cm in df['compound']:
			if cm not in self.compounds:
				self.compounds[cm] = Compound(cm)

		def _ad(s):
			if not np.isnan(s['areal_density']):
				return s['areal_density']
			if not np.isnan(s['mass']) and not np.isnan(s['area']):
				return 1E3*s['mass']/s['area']
			if not np.isnan(s['thickness']):
				if not np.isnan(s['density']):
					return 1E2*s['density']*s['thickness']
				else:
					return 1E2*self.compounds[s['compound']].density*s['thickness']

		ad = df.apply(_ad, axis=1)

		self.stack = pd.DataFrame({'name':df['name'], 'compound':df['compound'], 'areal_density':ad})[['name', 'compound', 'areal_density']]
		self._solve()


	def _filter_cols(self, df):
		cols = []
		for cl in df.columns:
			c = cl.lower()
			if c=='cm':
				cols.append('compound')
			elif c=='d':
				cols.append('density')
			elif c=='t':
				cols.append('thickness')
			elif c=='m':
				cols.append('mass')
			elif c=='a':
				cols.append('area')
			elif c=='ad':
				cols.append('areal_density')
			elif c=='nm':
				cols.append('name')
			else:
				cols.append(c)

		df.columns = cols
		for c in ['name','density','thickness','mass','area','areal_density']:
			if c not in df.columns:
				df[c] = np.nan
		return df


	def _parse_kwargs(self, **kwargs):
		self._dE0 = float(kwargs['dE0']) if 'dE0' in kwargs else 0.01*self._E0
		self._N = int(kwargs['N']) if 'N' in kwargs else 10000
		self._dp = float(kwargs['dp']) if 'dp' in kwargs else 1.0
		self._chunk_size = int(kwargs['chunk_size']) if 'chunk_size' in kwargs else int(1E7)
		self._accuracy = float(kwargs['accuracy']) if 'accuracy' in kwargs else 0.01
		self._min_steps = int(kwargs['min_steps']) if 'min_steps' in kwargs else 2
		self._max_steps = int(kwargs['max_steps']) if 'max_steps' in kwargs else 50

		if 'compounds' in kwargs:
			compounds = kwargs['compounds']
			if type(compounds)==str:
				if compounds.endswith('.json'):
					js = json.loads(open(compounds).read())
					cms = [i for i in js]
					self.compounds = {cm:Compound(cm, weights=js[cm]) for cm in cms}

				elif compounds.endswith('.csv'):
					df = pd.read_csv(filename, header=0).fillna(method='ffill')
					cms = [str(i) for i in pd.unique(df['compound'])]
					self.compounds = {cm:Compound(cm, weights=df[df['compound']==cm]) for cm in cms}

				elif compounds.endswith('.db'):
					df = pd.read_sql('SELECT * FROM compounds', _get_connection(filename))
					cms = [str(i) for i in pd.unique(df['compound'])]
					self.compounds = {cm:Compound(cm, weights=df[df['compound']==cm]) for cm in cms}


			elif type(compounds)==pd.DataFrame:
				cms = [str(i) for i in pd.unique(compounds['compound'])]
				self.compounds = {cm:Compound(cm, weights=compounds[compounds['compound']==cm]) for cm in cms}

			elif type(compounds)==list:
				for cm in compounds:
					if type(cm)==Compound or type(cm)==Element:
						### ci.Compound class
						self.compounds[cm.name] = cm

			elif type(compounds)==dict:
				### e.g. {'Water':{'H':2, 'O':1}, 'Brass':{'Cu':-66,'Zn':-33}}
				self.compounds = {c:Compound(c, weights=compounds[c]) for c in compounds}



	def _solve_chunk(self, N):
		E0 = self._E0+self._dE0*np.random.normal(size=int(N))
		bins = np.arange(0.0, self._E0+10.0*self._dE0, min([0.1, self._E0/500.0]))

		hists = []
		dp = self._dp

		for n, sm in self.stack.iterrows():
			E_bar = [E0]

			if np.average(E0)<=0.0:
				hists.append(np.concatenate([[N],np.zeros(len(bins)-2)]))

			else:
				cm = self.compounds[sm['compound']]
				ad = sm['areal_density']
				steps = int((1.0/self._accuracy)*ad*dp*cm.S(np.average(E0), particle=self._particle, density=1E-3)/np.average(E0))
				steps = min([max([self._min_steps, steps]), self._max_steps])

				dr = (1.0/float(steps))
				for i in range(steps):
					S1 = cm.S(E0, particle=self._particle, density=1E-3)

					E1 = E0 - dr*dp*ad*S1
					E1 = np.where(E1>0, E1, 0.0)

					E0 = E0 - dr*0.5*dp*ad*(S1+cm.S(E1, particle=self._particle, density=1E-3))
					E0 = np.where(E0>0, E0, 0.0)

					E_bar.append(E0)

				hists.append(np.histogram(np.concatenate(E_bar), bins=bins)[0])

		return hists

	def _solve(self):
		dN = np.linspace(0, self._N, int(np.ceil(self._N/float(self._chunk_size)))+1, dtype=int)
		bins = np.arange(0.0, self._E0+10.0*self._dE0, min([0.1, self._E0/500.0]))
		energy = 0.5*(bins[1:]+bins[:-1])

		histos = list(map(self._solve_chunk, dN[1:]-dN[:-1]))

		self._flux_list = []
		warn = True
		mu_E, sig_E = [], []

		for n in range(len(self.stack)):
			sm = {}
			sm['flux'] = np.sum([h[n] for h in histos], axis=0)
			sm['flux'] = sm['flux']/np.sum(sm['flux'])

			mu_E.append(np.sum(sm['flux']*energy))
			sig_E.append(np.sqrt(np.sum(sm['flux']*(energy-mu_E[-1])**2)))

			lh = np.where(sm['flux']>0)[0]
			if lh.size:
				if lh[0]==0 and warn:
					print('WARNING: Beam stopped in foil {}'.format(n))
					warn = False

			if str(self.stack['name'][n])!='nan':
				sm['name'] = str(self.stack['name'][n])
				sm['flux'] = sm['flux'][lh[0]:lh[-1]+1]
				sm['energy'] = energy[lh[0]:lh[-1]+1]
				sm['bins'] = bins[lh[0]:lh[-1]+2]
				self._flux_list.append(sm)

		cols = ['name','energy','flux']
		self.fluxes = pd.concat([pd.DataFrame({c:sm[c] for c in cols}, columns=cols) for sm in self._flux_list], ignore_index=True)

		self.stack['mu_E'] = mu_E
		self.stack['sig_E'] = sig_E


	def _filter_samples(self, df, name):
		if name==False:
			return df

		df_NN = df[df['name'].notnull()]
		if name==True:
			return df_NN

		match = [bool(re.search(name, nm)) for nm in df_NN['name']]
		return df_NN[match]

	def saveas(self, filename, save_fluxes=True, filter_name=True):
		""" Description

		...

		Parameters
		----------
		filename : str
			Description of x

		save_fluxes : bool, optional
			Description

		filter_name : bool or str, optional
			Description

		Examples
		--------

		"""

		if any([filename.endswith(e) for e in ['.png','.pdf','.eps','.pgf','.ps','.raw','.rgba','.svg','.svgz']]):
			self.plot(saveas=filename, show=False)

		if filename.endswith('.csv'):
			self._filter_samples(self.stack, filter_name).to_csv(filename, index=False)
			if save_fluxes:
				self._filter_samples(self.fluxes, filter_name).to_csv(filename.replace('.csv','_fluxes.csv'), index=False)

		if filename.endswith('.db'):
			self._filter_samples(self.stack, filter_name).to_sql('stack', _get_connection(filename), if_exists='replace', index=False)
			if save_fluxes:
				self._filter_samples(self.fluxes, filter_name).to_sql('fluxes', _get_connection(filename), if_exists='replace', index=False)

		if filename.endswith('.json'):
			json.dump(json.loads(self._filter_samples(self.stack, filter_name).to_json(orient='records')), open(filename, 'w'), indent=4)
			if save_fluxes:
				json.dump(json.loads(self._filter_samples(self.fluxes, filter_name).to_json(orient='records')), open(filename.replace('.json','_fluxes.json'), 'w'), indent=4)

		
	def summarize(self, filter_name=True):
		""" Description

		...

		Parameters
		----------
		filter_name : bool or str, optional
			Description of x

		Examples
		--------

		"""

		m = {}
		for n,sm in self._filter_samples(self.stack, filter_name).iterrows():
			if sm['name'] is None:
				if sm['compound'] not in m:
					m[sm['compound']] = 1
				nm = sm['compound']+'--'+str(m[sm['compound']])
				m[sm['compound']] += 1

			else:
				nm = sm['name']
			print(nm+': '+str(round(sm['mu_E'], 2))+' +/- '+str(round(sm['sig_E'], 2))+' (MeV)')

		
	def plot(self, filter_name=True, **kwargs):
		""" Description

		...

		Parameters
		----------
		filter_name : bool or str, optional
			Description of x

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------

		"""

		f,ax = _init_plot(**kwargs)
		for sm in self._flux_list:
			if type(filter_name)==str:
				if not re.search(filter_name, sm['name']):
					continue

			eng = np.array([sm['bins'][:-1], sm['bins'][1:]]).T.flatten()
			phi = np.array([sm['flux'], sm['flux']]).T.flatten()
			ax.plot(eng, phi, label=sm['name'])

		ax.set_xlabel('Energy (MeV)')
		ax.set_ylabel('Flux (a.u.)')
		ax.legend(loc=0)
		return _draw_plot(f, ax, **kwargs)
		