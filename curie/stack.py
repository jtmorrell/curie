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
	"""Stack

	...
	
	Parameters
	----------
	stack : list of dicts or str
		Description of parameter `x`.

	particle : str
		Description

	E0 : float
		Description


	Other Parameters
	----------------
	compounds : str, pandas.DataFrame, list or dict
		Description

	dE0 : float
		Description

	N : int
		Description

	dp : float
		Description

	chunk_size : int
		Description

	accuracy : float
		Description

	min_steps : int
		Description

	max_steps : int
		Description

	Attributes
	----------
	stack : pandas.DataFrame
		Description

	fluxes : pandas.DataFrame
		Description

	compounds : dict
		Description

	Examples
	--------

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
			elif c=='rho':
				cols.append('density')
			elif c=='r':
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
					df = pd.read_csv(filename, header=0, names=['compound', 'element', 'weight']).fillna(method='ffill')
					cms = [str(i) for i in pd.unique(df['compound'])]
					self.compounds = {cm:Compound(cm, weights=df[df['compound']==cm]) for cm in cms}

			elif type(compounds)==pd.DataFrame:
				cms = [str(i) for i in pd.unique(compounds['compound'])]
				self.compounds = {cm:Compound(cm, weights=compounds[compounds['compound']==cm]) for cm in cms}

			elif type(compounds)==list:

				for cm in compounds:
					if type(cm)==str:
						### e.g. ['H20', 'RbCl']
						self.compounds[cm] = Compound(cm)

					elif type(cm)==Compound or type(cm)==Element:
						### ci.Compound class
						self.compounds[cm.name] = cm

					elif type(cm)==dict:
						### e.g. [{'cm':'Water','weights':{'H':2,'O':1}}]
						if 'cm' in cm:
							c = cm['cm']
						elif 'compound' in cm:
							c = cm['compound']
						self.compounds[c] = Compound(c, weights=cm['weights'])

			elif type(compounds)==dict:
				### e.g. {'Water':{'H':2, 'O':1}, 'RbCl':{'Rb':1,'Cl':1}}
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
		