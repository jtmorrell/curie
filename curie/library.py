from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd

from .data import _get_connection
from .isotope import Isotope

class Library(object):
	"""Library

	...
	
	Parameters
	----------
	x : type
		Description of parameter `x`.

	Examples
	--------

	"""

	def __init__(self, name):
		name = name.lower()
		if name in ['endf']:
			self.db_name = 'endf'
		elif name in ['tendl']:
			self.db_name = 'tendl'
		elif name in ['tendl_n_rp','tendl_nrp','tendl_n','nrp','rpn']:
			self.db_name = 'tendl_n_rp'
		elif name in ['tendl_p_rp','tendl_prp','tendl_p','prp','rpp']:
			self.db_name = 'tendl_p_rp'
		elif name in ['tendl_d_rp','tendl_drp','tendl_d','drp','rpd']:
			self.db_name = 'tendl_d_rp'
		elif name in ['irdff']:
			self.db_name = 'irdff'
		elif name in ['iaea','iaea-cpr','iaea-monitor','cpr','iaea_cpr','iaea_monitor','medical','iaea-medical','iaea_medical']:
			self.db_name = 'iaea_medical'
		else:
			raise ValueError('Library {} not recognized.'.format(name))
			
		self._con = _get_connection(self.db_name)
		self.name = {'endf':'ENDF/B-VII.1','tendl':'TENDL-2015','irdff':'IRDFF-II','iaea':'IAEA CP-Reference (2017)'}[self.db_name.split('_')[0]]
		self._warn = True

	def __str__(self):
		return self.name
		

	def search(self, target=None, incident=None, outgoing=None, product=None, _label=False):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		if incident is not None:
			incident = incident.lower()

		if outgoing is not None:
			outgoing = outgoing.lower()

		if target is not None:
			if '-' in target:
				if target.endswith('m') or target.endswith('g'):
					target = Isotope(target).name
				else:
					target = Isotope(target)._short_name

		if product is not None:
			if '-' in product:
				if product.endswith('m') or product.endswith('g'):
					product = Isotope(product).name
				else:
					product = Isotope(product)._short_name

		ss = 'SELECT * FROM all_reactions'

		if self.db_name in ['endf','tendl','irdff']:
			if incident is not None:
				if incident!='n':
					return []
			q = [(i+'%' if not n%2 else i) for n,i in enumerate([target, outgoing, product]) if i]
			ss += ' WHERE ' if len(q) else ''
			ss += ' AND '.join([i for i in [('target LIKE ?' if target else ''),('outgoing=?' if outgoing else ''),('product LIKE ?' if product else '')] if i])
			reacs = [r.to_list() for n,r in pd.read_sql(ss, self._con, params=tuple(q)).iterrows()]
			fmt = '{0}(n,{1}){2}'

		elif self.db_name in ['tendl_n_rp','tendl_p_rp','tendl_d_rp']:
			if incident is not None:
				if incident!=self.db_name.split('_')[1]:
					return []

			if product:
				if 'm' not in product and 'g' not in product:
					product += 'g'
					if self._warn:
						print('WARNING: Product isomeric state not specified, ground state assumed.')
						self._warn = False

			q = [i+'%' for i in [target, product] if i]
			ss += ' WHERE ' if len(q) else ''
			ss += ' AND '.join([i for i in [('target LIKE ?' if target else ''),('product LIKE ?' if product else '')] if i])
			reacs = [r.to_list() for n,r in pd.read_sql(ss, self._con, params=tuple(q)).iterrows()]
			fmt = '{0}('+self.db_name.split('_')[1]+',x){1}'

		elif self.db_name=='iaea_medical':
			q = [(i+'%' if n in [0,3] else i) for n,i in enumerate([target, incident, outgoing, product]) if i]
			ss += ' WHERE ' if len(q) else ''
			ss += ' AND '.join([i for i in [('target LIKE ?' if target else ''),('incident=?' if incident else ''),('outgoing=?' if outgoing else ''),('product LIKE ?' if product else '')] if i])
			reacs = [r.to_list() for n,r in pd.read_sql(ss, self._con, params=tuple(q)).iterrows()]
			fmt = '{0}({1},{2}){3}'

		if target:
			reacs = [i for i in reacs if i[0].lower()==target.lower()]
		if _label:
			return [i[-1] for i in reacs]

		return [fmt.format(*i) for i in reacs]

		
	def retrieve(self, target=None, incident=None, outgoing=None, product=None):
		""" Description

		...

		Parameters
		----------
		x : type
			Description of x

		Examples
		--------

		"""

		labels = self.search(target, incident, outgoing, product, _label=True)

		if not len(labels)==1:
			raise ValueError('{0}({1},{2}){3}'.format(target, incident, outgoing, product)+' is not a unique reaction.')

		if not target:
			raise ValueError('Target Must be specified.')

		if '-' in target:
			if target.endswith('m') or target.endswith('g'):
				target = Isotope(target).name
			else:
				target = Isotope(target)._short_name

		if self.db_name in ['endf','tendl','tendl_n_rp','tendl_p_rp','tendl_d_rp']:
			table = ''.join(re.findall('[A-Z]+', target))+'_'+''.join(re.findall('[0-9]+', target))+('m' if 'm' in target else '')

			return pd.read_sql('SELECT energy,{0} FROM {1}'.format(labels[0], table), self._con).to_numpy()*(np.array([1E-6, 1E3]) if self.db_name=='endf' else np.ones(2))

		elif self.db_name=='irdff':
			return pd.read_sql('SELECT * FROM {}'.format(labels[0]), self._con).to_numpy()*np.array([1E-6, 1E3, 1E3])

		elif self.db_name=='iaea_medical':
			if incident is None:
				raise ValueError('Incident particle must be specified.')
			table = {'n':'neutron','p':'proton','d':'deuteron','h':'helion','a':'alpha','g':'gamma'}[incident]
			return pd.read_sql('SELECT energy,cross_section,unc_cross_section FROM {0} WHERE target LIKE {1} AND product={2}'.format(table, '%'+target+'%', labels[0]), self._con).to_numpy()
		