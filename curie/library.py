from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import pandas as pd

from .data import _get_connection
from .isotope import Isotope

class Library(object):
	"""Library of nuclear reaction data

	Provides a means of searching and retrieving data from various nuclear reaction
	libraries.  The currently available libraries are ENDF/B-VII.1, TENDL-2015, IRDFF-II,
	and the IAEA Medical Monitor reaction library.  For neutrons, the libraries are
	categorized by either the exlusive reaction, as in ENDF, TENDL, and partially IRDFF,
	or by the residual product (rp), in TENDL and partially IRDFF.  For charged particles,
	all libraries (TENDL, IAEA) are categorized by the residual product.  As such, TENDL
	is split into separate libraries based on incident particle, and by exclusive reaction
	type or residual product.
	
	Parameters
	----------
	name : str
		The name of the library.  For exclusive neutron reactions, use 'endf', 'tendl', 
		or 'irdff'.  For residual product neutron reactions, use 'tendl_n' or 'irdff'.  Use
		'tendl_p' for proton reactions, 'tendl_d' for deuteron reactions, or 'iaea' for either.

	Examples
	--------
	>>> lb = ci.Library('tendl_n')
	>>> print(lb.name)
	TENDL-2015
	>>> lb = ci.Library('endf')
	>>> print(lb.name)
	ENDF/B-VII.1

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
		"""Searches the library for reactions

		Returns a list of reactions that match the specified target, incident
		or outgoing projectile, or product.  Any combination of arguments can
		be specified, but at least one must be specified.  Note that if the library
		is a TENDL residual-product library, outgoing does not need to be specified.

		Parameters
		----------
		target : str, optional
			The target nucleus.  Some libraries support natural elements, e.g. 'natEl'.

		incident : str, optional
			Incident particle.  Must be one of 'n', 'p' or 'd'.  Only needed for IAEA
			library with multiple incident projectiles optional.

		outgoing : str, optional
			Outgoing particle, or reaction shorthand.  E.g. '2n', 'd', 'f', 'inl', 'x'.
			Does not need to be specified for TENDL residual product libraries.

		product : str, optional
			The product isotope.  Not required for some exclusive reactions, e.g. 
			'115IN(n,g)116IN' is the same as '115IN(n,g)', but not the same as
			'115IN(n,g)116INm1'.

		Returns
		-------
		available_reactions : list of str (reaction names)
			Reactions that were found in the libary matching the search criteria.

		Examples
		--------
		>>> lb = ci.Library('tendl_p')
		>>> print(lb.search(target='Sr-86', product='Y-86g'))
		['86SR(p,x)86Yg']
		>>> lb = ci.Library('endf')
		>>> print(lb.search(target='226RA', product='225RA'))
		['226RA(n,2n)225RA']

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
		"""Pull reaction data from the library

		Returns a numpy ndarray containing the reaction energy grid (in MeV),
		the cross section (in mb), and for IRDFF and IAEA, the uncertainty in the 
		cross section (mb).  The arguments are the same as for `Library.search()`.

		Parameters
		----------
		target : str, optional
			The target nucleus.  Some libraries support natural elements, e.g. 'natEl'.

		incident : str, optional
			Incident particle.  Must be one of 'n', 'p' or 'd'.  Only needed for IAEA
			library with multiple incident projectiles optional.

		outgoing : str, optional
			Outgoing particle, or reaction shorthand.  E.g. '2n', 'd', 'f', 'inl', 'x'.
			Does not need to be specified for TENDL residual product libraries.

		product : str, optional
			The product isotope.  Not required for some exclusive reactions, e.g. 
			'115IN(n,g)116IN' is the same as '115IN(n,g)', but not the same as
			'115IN(n,g)116INm1'.

		Returns
		-------
		reaction : np.ndarray
			Numpy ndarray containing the reaction data. Energy is in MeV, cross section
			is in mb.  The first column is energy, the second column is cross section, and
			if the library provides uncertainties in cross section this will be in the third column.

		Examples
		--------
		>>> lb = ci.Library('endf')
		>>> print(lb.retrieve(target='226RA', product='225RA')[-8:])
		[[18.5      11.97908 ]
		 [18.75     10.30011 ]
		 [19.        8.62115 ]
		 [19.25      6.942188]
		 [19.5       5.263225]
		 [19.75      3.584263]
		 [19.875     2.744781]
		 [20.        1.9053  ]]

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
		