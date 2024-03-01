from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from x4i3 import exfor_manager, exfor_entry, exfor_exceptions
import re
import matplotlib.pyplot as plt


from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from .isotope import Isotope
from .library import Library
from .element import Element


class Reaction(object):
	"""Cross section data for nuclear reactions

	Contains reaction cross sections as a function of incident energy,
	and some useful methods for manipulating cross section data, such as
	flux-averages, integrated cross-sections, and interpolation.  All 
	cross sections (and uncertainties) are in mb, and all energies are
	in MeV.
	
	Parameters
	----------
	reaction_name : str
		Name of the reaction, in nuclear reaction notation. E.g. '115IN(n,g)',
		'235U(n,f)', '139LA(p,x)134CE', 'Ra-226(n,2n)Ra-225', 'Al-27(n,a)', etc.

	library : str, optional
		Name of the library to use, or 'best' (default).

	Attributes
	----------
	target : str
		The target nucleus.  Some libraries support natural elements, e.g. 'natEl'. 

	incident : str
		Incident particle. E.g. 'n', 'p', 'd'. 

	outgoing : str
		Outgoing particle, or reaction shorthand.  E.g. '2n', 'd', 'f', 'inl', 'x'.
		Will always be 'x' for (TENDL) residual product libraries. 

	product : str
		The product isotope. 

	eng : np.ndarray
		Incident particle energy, in MeV. 

	xs : np.ndarray
		Reaction cross section, in mb. 

	unc_xs : np.ndarray
		Uncertainty in the cross section, in mb.  If not provided by the
		library, default is zeros of same shape as xs. 

	name : str
		Name of the reaction in nuclear reaction notation. 

	library : ci.Library
		Nuclear reaction library.  printing `rx.library.name` will give the
		name of the library. 

	TeX : str
		LaTeX formatted reaction name. 


	Examples
	--------
	>>> rx = ci.Reaction('226RA(n,2n)')
	>>> print(rx.library.name)
	ENDF/B-VII.1
	>>> rx = ci.Reaction('226RA(n,x)225RA')
	>>> print(rx.library.name)
	TENDL-2015
	>>> rx = ci.Reaction('115IN(n,inl)')
	>>> print(rx.library.name)
	IRDFF-II

	"""

	def __init__(self, reaction_name, library='best'):
		self.target, p = tuple(reaction_name.split('('))
		p, self.product = tuple(p.split(')'))
		self.incident, self.outgoing = tuple(p.split(','))
		self.incident, self.outgoing = self.incident.lower(), self.outgoing.lower()
		self._rx = [self.target, self.incident, self.outgoing, self.product]
		self.name = reaction_name
		self.exfor_target, self.exfor_reaction, self.exfor_product = self.curie_to_exfor()
		self.plot_tendl = False
		self.multiple_product_subentries = False

		if library.lower()=='best':
			if self.incident=='n':
				for lb in ['irdff','endf','iaea','tendl','tendl_n_rp']:
					self.library = Library(lb)
					if lb=='tendl_n_rp':
						self._check(True)
					elif self._check():
						break
			elif self.incident in ['p','d']:
				for lb in ['iaea','tendl_'+self.incident+'_rp']:
					self.library = Library(lb)
					if lb=='tendl_d_rp':
						self._check(True)
					elif self._check():
						break
			else:
				self.library = Library('iaea')
				self._check(True)
		else:
			self.library = Library(library)
			self._check(True)

		if '*' not in self.name:
			self.name = self.library.search(*self._rx)[0]
			q = self.library.retrieve(*self._rx)
			self.eng = q[:,0]
			self.xs = q[:,1]
			if q.shape[1]==3:
				self.unc_xs = q[:,2]
			else:
				self.unc_xs = np.zeros(len(self.xs))
			self._interp = None
			self._interp_unc = None

		try:
			if 'nat' not in self.target:
				tg = Isotope(self.target).TeX
			else:
				tg = r'$^{nat}$'+self.target[3:].title()
			prd = Isotope(self.product).TeX if self.product else ''

			self.TeX = '{0}({1},{2}){3}'.format(tg, self.incident, self.outgoing, prd)

		except:
			self.TeX = reaction_name

	def _check(self, err=False):
		c = len(self.library.search(*self._rx))==1
		if err and not c:
			raise ValueError('Reaction '+self.name+' not found or not unique.')
		return c

	def __str__(self):
		return self.name
		
	def interpolate(self, energy):
		"""Interpolated cross section

		Linear interpolation of the reaction cross section along the
		input energy grid.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		Returns
		-------
		cross_section : np.ndarray
			Interpolated cross section, in mb.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)', 'IRDFF')
		>>> print(rx.interpolate(0.5))
		161.41656650941306
		>>> print(rx.interpolate([0.5, 1.0, 5.0]))
		[161.41646651 171.81486757 8.8822]

		""" 

		if self._interp is None:
			kind = 'linear'
			fv = 'extrapolate'
			i = 0
			if self.library.name.lower().startswith('tendl'):
				kind = 'quadratic'
				fv = 0.0
				ix = np.where(self.xs>0)[0]
				if len(ix)>0:
					i = max((ix[0]-1, 0))
					if len(self.xs)-i<5:
						kind = 'linear'
			self._interp = interp1d(self.eng[i:], self.xs[i:], bounds_error=False, fill_value=fv, kind=kind)
		_interp = self._interp(energy)
		return np.where(_interp>0, _interp, 0.0)

	def interpolate_unc(self, energy):
		"""Uncertainty in interpolated cross section

		Linear interpolation of the uncertainty in the reaction cross section
		along the input energy grid, for libraries where uncertainties are provided.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		Returns
		-------
		unc_cross_section : np.ndarray
			Uncertainty in the interpolated cross section, in mb.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)', 'IRDFF')
		>>> print(rx.interpolate_unc(0.5))
		3.9542683715745546
		>>> print(rx.interpolate_unc([0.5, 1.0, 5.0]))
		[3.95426837 5.88023936 0.4654]

		""" 

		if self._interp_unc is None:
			self._interp_unc = interp1d(self.eng, self.unc_xs, bounds_error=False, fill_value=0.0)
		return self._interp_unc(energy)
		
	def integrate(self, energy, flux, unc=False):
		"""Reaction flux integral

		Integrate the product of the cross section and flux along the input energy grid.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		flux : array_like
			Incident particle flux as a function of the input energy grid.

		unc : bool, optional
			If `True`, returns the both the flux integral and the uncertainty. If `False`,
			just the flux integral is returned. Default `False`.

		Returns
		-------
		xs_integral : float or tuple
			Reaction flux integral if `unc=False` (default), or reaction flux integral
			and uncertainty, if `unc=True`.

		Examples
		--------
		>>> x = ci.Reaction('Ni-58(n,p)')
		>>> eng = np.linspace(1, 5, 20)
		>>> phi = np.ones(20)
		>>> print(rx.integrate(eng, phi))
		833.4435915974148
		>>> print(rx.integrate(eng, phi, unc=True))
		(833.4435915974148, 19.91851943674977)

		"""

		E = np.asarray(energy)
		phisig = np.asarray(flux)*self.interpolate(E)
		if unc:
			unc_phisig = np.asarray(flux)*self.interpolate_unc(E)
			return np.sum(0.5*(E[1:]-E[:-1])*(phisig[:-1]+phisig[1:])), np.sum(0.5*(E[1:]-E[:-1])*(unc_phisig[:-1]+unc_phisig[1:]))
		return np.sum(0.5*(E[1:]-E[:-1])*(phisig[:-1]+phisig[1:]))
		
	def average(self, energy, flux, unc=False):
		"""Flux averaged reaction cross section

		Calculates the flux-weighted average reaction cross section, using the
		input flux and energy grid.

		Parameters
		----------
		energy : array_like
			Incident particle energy, in MeV.

		flux : array_like
			Incident particle flux as a function of the input energy grid.

		unc : bool, optional
			If `True`, returns the both the flux average cross section and the uncertainty. If `False`,
			just the average cross section is returned. Default `False`.

		Returns
		-------
		average_xs : float or tuple
			Flux-averaged reaction cross section if `unc=False` (default), or average
			and uncertainty, if `unc=True`.

		Examples
		--------
		>>> rx = ci.Reaction('Ni-58(n,p)')
		>>> eng = np.linspace(1, 5, 20)
		>>> phi = np.ones(20)
		>>> print(rx.average(eng, phi))
		208.3608978993537
		>>> print(rx.average(eng, phi, unc=True))
		(208.3608978993537, 4.979629859187442)

		"""

		E, phi = np.asarray(energy), np.asarray(flux)
		phisig = phi*self.interpolate(E)
		dE = E[1:]-E[:-1]
		if unc:
			unc_phisig = np.asarray(flux)*self.interpolate_unc(E)
			return np.sum(0.5*dE*(phisig[:-1]+phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:])), np.sum(0.5*dE*(unc_phisig[:-1]+unc_phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:]))
		return np.sum(0.5*dE*(phisig[:-1]+phisig[1:]))/np.sum(0.5*dE*(phi[:-1]+phi[1:]))
		
	def plot(self, energy=None, label='reaction', title=False, **kwargs):
		"""Plot the cross section

		Plots the energy differential cross section.

		Parameters
		----------
		energy : array_like, optional
			Energy grid along which to plot the cross section.  If None, the
			energy grid provided by the library will be used. 

		label : str, optional
			Axes label.  If label='reaction', the label will be the reaction name.
			If 'library', it will be the name of the cross section library.
			If 'both', then the reaction name and library will be given.  If
			none of these options, pyplot will be called with `ax.plot(..., label=label)`.

		title : bool, optional
			Display the reaction name as the plot title.  Default, False.

		Other Parameters
		----------------
		**kwargs
			Optional keyword arguments for plotting.  See the 
			plotting section of the curie API for a complete
			list of kwargs.

		Examples
		--------
		>>> rx = ci.Reaction('115IN(n,g)')
		>>> rx.plot(scale='loglog')
		>>> rx = ci.Reaction('35CL(n,p)')
		>>> f, ax = rx.plot(return_plot=True)
		>>> rx = ci.Reaction('35CL(n,el)')
		>>> rx.plot(f=f, ax=ax, scale='loglog')

		"""

		f, ax = _init_plot(**kwargs)

		if title:
			ax.set_title(self.TeX)

		if label is not None:
			if label.lower() in ['both','library','reaction']:
				label = {'both':'{0}\n({1})'.format(self.TeX, self.library.name),'library':self.library.name,'reaction':self.TeX}[label.lower()]

		unc_xs = None
		if energy is None:
			if self.library.name.lower().startswith('tendl'):
				eng = np.linspace(min(self.eng), max(self.eng), 801)
				xs = self.interpolate(eng)
			else:
				eng, xs = self.eng, self.xs
				if np.any(self.unc_xs>0):
					unc_xs = self.unc_xs
		else:
			eng, xs = np.asarray(energy), self.interpolate(energy)
			ux = self.interpolate_unc(energy)
			if np.any(ux>0):
				unc_xs = ux

		line, = ax.plot(eng, xs, label=label)
		if unc_xs is not None:
			ax.fill_between(eng, xs+unc_xs, xs-unc_xs, facecolor=line.get_color(), alpha=0.5)

		if self.library.name.lower().startswith('tendl'):
			wh = np.where((self.eng>=min(eng))&(self.eng<=max(eng)))
			elib = self.eng[wh]
			xslib = self.xs[wh]
			ax.plot(elib, xslib, ls='None', marker='o', color=line.get_color())

		ax.set_xlabel('Incident Energy (MeV)')
		ax.set_ylabel('Cross Section (mb)')

		if label:
			ax.legend(loc=0)

		return _draw_plot(f, ax, **kwargs)
	

	def curie_to_exfor(self):
		parsed_target = self.name.split('(')[0].strip('g')
		reaction_code = self.name.split('(')[1].split(')')[0].strip('g')
		parsed_product = self.name.split('(')[1].split(')')[1].strip('g')

		# print(parsed_target)

		# target = ''

		# convert natural abundance notation to EXFOR
		if 'nat' in parsed_target:
			# Monoisotopic elements:    4-BE-9 27-CO-59 59-PR-141, 
										# 9-F-19 33-AS-75 65-TB-159
										# 11-NA-23 39-Y-89 67-HO-165
										# 13-AL-27 41-NB-93 69-TM-169
										# 15-P-31 45-RH-103 79-AU-197
										# 21-SC-45 53-I-127 83-BI-209
										# 25-MN-55 55-CS-133 90-TH-232
			# Check for nearly monoisotopic target, may be inconsistent based on EXFOR compiler
			element_symbol = parsed_target.strip('nat')
			print (element_symbol)
			nearly_monoisotopic_elements = ['H', 'N', 'LA', 'HE', 'O', 'TA', 'C', 'V']
			if element_symbol in nearly_monoisotopic_elements:
				nearly_monoisotopic_A = [1, 14, 139, 4, 16, 181, 12, 51]
				index = nearly_monoisotopic_elements.index(element_symbol)
				print('Element',element_symbol.capitalize(),'is nearly monoisotopic, and may be listed in EXFOR as either '+element_symbol+'-0 or '+element_symbol+'-'+str(nearly_monoisotopic_A[index])+', please double-check the returned results!')
			parsed_target = parsed_target.replace('nat','0')



		# Look for ZZZ-* (any isotope) targets
		if '*' in parsed_target:
			# print('found!')
			self.exfor_target = parsed_target.replace('*','9999')
			# substrings = re.split('(\D+)',exfor_target)
			# # exfor_target = (substrings[1]+'-'+substrings[0]).strip(' ').upper()
			# exfor_target = (substrings[1]+'-'+substrings[0]).upper()
		else:
			substrings = re.split('(\D+)',parsed_target)
			# exfor_target = (substrings[1]+'-'+substrings[0]).strip(' ').upper()
			self.exfor_target = (substrings[1]+'-'+substrings[0]).upper()


		# Look for ZZZ-* (any isotope) products
		if '*' in parsed_product:
			# print('found!')
			self.exfor_product = parsed_product.replace('*','9999')
		else:
			substrings = re.split('(\D+)',parsed_product)
			self.exfor_product = (substrings[1]+'-'+substrings[0]).upper()

		self.exfor_reaction = reaction_code.upper().replace('X','*')






		print(self.exfor_target)
		# print(exfor_rxn)
		# print(exfor_product)

		# print('target: '+parsed_target)
		# print('rxn: '+reaction_code)
		# print('product: '+parsed_product)

		return self.exfor_target, self.exfor_reaction, self.exfor_product




	# ---------------------------------------------------------------


	def search_exfor(self, plot_results=False, plot_tendl=False):
		# print(self.name)
		# self.exfor_target, self.exfor_rxn, self.exfor_product = self.curie_to_exfor()
		db = exfor_manager.X4DBManagerDefault()
		self.target_element = self.exfor_target.split('-')[0].capitalize()
		multiple_product_subentries = False
		

		# Check for *-products (aka A=9999)
		# print('PRODUCT:',product)
		if '9999' in self.exfor_product:
			# title_product = product.replace('9999','*')
			self.exfor_product = self.exfor_product.strip('9999')
			# Avoid a spaghetti plot...
			self.plot_tendl = False
			multiple_product_subentries = True
		
		# Check for *-targets (aka A=9999)
		# print('TARGET:',target)
		if '9999' in self.exfor_target:
			self.exfor_target = self.exfor_target.strip('9999')
			# Avoid a spaghetti plot...
			self.plot_tendl = False
			multiple_product_subentries = True
		else:
			# Detect if target material is natural abundance
			if self.exfor_target.split('-')[1] == 0:
				self.enriched = True
			else:
				self.enriched = False

		print('PRODUCT:',self.exfor_product)


		x = db.retrieve(target=self.exfor_target,reaction=self.exfor_reaction,quantity='SIG' )
		# x = db.retrieve(target='LA-0',reaction=reaction,quantity='SIG' )
		# Wildcards in 'reaction' seem to produce a TON of false positives, even when querying with 'product'
		# x = db.retrieve(target='BI-209',reaction='D,*',product='PO-209',quantity='SIG' )
		# print(x.keys())
		# query() just searches for data matching search parameters
		# print(db.query(target='PU-239',reaction='N,2N',quantity='SIG',author='Lougheed' ))
		# print(db.query(target='LA-*',reaction=self.exfor_reaction,quantity='SIG' ))

		


		# Hold extracted data for plotting
		plot_Dict = {}
		for key in x.keys():
			entry = x[key]
			# print(entry)

			datasets = entry.getDataSets()

			# Find columns for relevant data
			energy_col = -1
			unc_energy_col = -1
			xs_col = -1
			unc_xs_col = -1
			xs_unit_scalar = 1
			energy_unit_scalar = 1

			# Make sure only one subentry is retrieved per entry!
			num_of_sub_subentries = len(list(datasets.keys()))

			# print(product)
			# print(target)

			subentry_list = []


			if num_of_sub_subentries == 1:
				# We're all good...
				# subentry = next(iter(datasets.values()))
				i = next(iter(datasets.values()))
				# print('Reaction: ',str(i.reaction))
				# print(i.subent)
				# Only select subentries leading to the specified product
				if self.exfor_product in str(i.reaction) and "NON" not in str(i.reaction) and "TTY" not in i.reaction[0].quantity and "RECOM" not in i.reaction[0].quantity and "EVAL" not in i.reaction[0].quantity:
					subentry_list.append(i)
					# print('Reaction: ',str(i.reaction))
				else:
					continue

			else:
				# Poorly-formatted EXFOR - multiple subentries for one entry
				print('Number of datasets found in entry', next(iter(datasets))[1][0:5], ': ', num_of_sub_subentries)
				# print('Other datasets in this entry: ',datasets.keys())
				# print(type(datasets))

				for i in datasets.values():
					# print('Reaction: ',str(i.reaction))
					# Only select subentries leading to the specified product
					# print(str(i.reaction))
					if self.exfor_product in str(i.reaction) and "NON" not in i.reaction[0].quantity and "TTY" not in i.reaction[0].quantity and "RECOM" not in i.reaction[0].quantity and "EVAL" not in i.reaction[0].quantity:
						subentry_list.append(i)
						# print('Reaction: ',str(i.reaction))
						# # print(datasets[i].data)
						# print(subentry.subent)
						# print(i.data)	

			

			# if 'subentry' not in locals():
			# 	continue
			if len(subentry_list) == 0:
				continue

			# print(len(subentry))
			# print(subentry_list)

			# Pull metadata and stash into dictionary
			# print(type(subentry.getSimplified()))
			# print(dir(subentry.getSimplified()))
			# print(subentry.getSimplified())
			# print(subentry.getSimplified().reaction)
			# print(type(subentry.getSimplified().reaction))
			# for s in subentry.getSimplified().reaction:
			# print(subentry.getSimplified().reaction[0].quantity)
			# print(s.quantity)
			# print (dir(s))

			for subentry in subentry_list:
				# print(subentry)
				# print(subentry.data)
				# print(dir(subentry))
				# print(str(subentry.reaction))
				# print('SFC' in str(subentry.reaction))

				try:

					# Make sure data isn't actually RECOM or TTY
					if "RECOM" in subentry.getSimplified().reaction[0].quantity:
						print('RECOM data found, skipping...')
						continue
					elif "TTY" in subentry.getSimplified().reaction[0].quantity:
						print('TTY data found, skipping...')
						continue
					elif "SFC" in str(subentry.reaction):
						print('SFC data found, skipping...')
						continue
				except exfor_exceptions.NoValuesGivenError:
					# print('test')
					# print(subentry.reaction)
					# ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
					# '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', 
					# '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slotnames__', '__slots__', 
					# '__str__', '__subclasshook__', '__weakref__', 'append', 'author', 'citation', 'coupled', 'csv', 'data', 'getSimplified', 
					# 'institute', 'labels', 'legend', 'monitor', 'numcols', 'numrows', 'pubType', 'reaction', 'reference', 'reprHeader', 
					# 'setData', 'simplified', 'sort', 'strHeader', 'subent', 'title', 'units', 'xmgraceHeader', 'year']
					continue
				

				# print(subentry.getSimplified().labels)
				
				for entry in subentry.getSimplified().labels:
					# print(entry)
					# Find what column the energy data are located in
					if entry in ("Energy", "EN", "EN-CM"): 
						energy_col = subentry.getSimplified().labels.index(entry)
						# print('Energy data found in column:',energy_col)
					# Find what column the xs data are located in
					if entry.casefold() == "Data".casefold(): 
						xs_col = subentry.getSimplified().labels.index(entry)
						# print('XS data found in column:',xs_col)
					if entry in ("d(Data)", "DATA-ERR"): 
						unc_xs_col = subentry.getSimplified().labels.index(entry)
						# print('XS data uncertainty found in column:',unc_xs_col)
					if entry in ("d(Energy)", "EN-ERR"): 
						unc_energy_col = subentry.getSimplified().labels.index(entry)
						# print('Energy data uncertainty found in column:',unc_energy_col)



				# ...grab the energy units
				# print(subentry.getSimplified().units[energy_col])
				if subentry.getSimplified().units[energy_col] in ("MEV", "MeV"):
					energy_unit_scalar = 1
				else:
					print("No energy units found for entry", next(iter(datasets))[1][0:5])

				# ...grab the xs units
				if subentry.getSimplified().units[xs_col].casefold() == "mb".casefold():
					xs_unit_scalar = 1
				elif subentry.getSimplified().units[xs_col].casefold() in ("barns".casefold(), "barn".casefold()):
					xs_unit_scalar = 1E3
				else:
					print("No XS units found for entry", next(iter(datasets))[1][0:5])

				# print(subentry.getSimplified().data)
				author_name = subentry.author[0].split('.',-1)[-1]
				year = subentry.year
				# print(author_name)
				# print(year)
				# print(subentry.subent)
				# print(type(subentry))
				plot_Dict[author_name+year+subentry.subent] = (author_name, # 0
										  year, # 1
										  np.array(subentry.getSimplified().data, dtype=float),  # 2
										  subentry.subent, # 3
										  energy_unit_scalar, # 4
										  xs_unit_scalar, # 5
										  energy_col, # 6
										  xs_col, # 7
										  unc_energy_col, # 8 
										  unc_xs_col,  # 9
										  subentry.reaction)  # 10

		print('---------------------------')
		# print(plot_Dict)


		if plot_results:
			self.plot_exfor(plot_Dict,self.plot_tendl,multiple_product_subentries)
		


	def plot_exfor(self, plot_Dict, plot_tendl=False, multiple_product_subentries=False):
		# plt.plot(tendl_data[:,0], tendl_data[:,1], label='TENDL-2021', color='k')
		# plt.plot(tendl_data_208[:,0], tendl_data_208[:,1], label='209', color='r')

		if len(plot_Dict) != 0:
			# Plot results
			k=0
			for index in plot_Dict:
				# Need to set up list of marker sizes to iterate over with k
				# print(k)
				# Use local variables
				author_name = plot_Dict[index][0]
				year = plot_Dict[index][1]
				plot_data = plot_Dict[index][2]
				subent = plot_Dict[index][3]
				energy_unit_scalar = plot_Dict[index][4]
				xs_unit_scalar = plot_Dict[index][5]
				energy_col = plot_Dict[index][6]
				xs_col = plot_Dict[index][7]
				unc_energy_col = plot_Dict[index][8]
				unc_xs_col = plot_Dict[index][9]
				subentry_reaction = plot_Dict[index][10]

				# print('subent ', subent)
				# print('energy_col ', energy_col)
				# print('xs_col ', xs_col)
				# print('unc_energy_col ', unc_energy_col)
				# print('unc_xs_col ', unc_xs_col)
				# print('author_name:', author_name)
				# print('plot_data ', plot_data)

				# print(dir(subentry_reaction[0]))
				# print(vars(subentry_reaction[0]))
				# print(subentry_reaction[0].getReactionType)
				# print(type(subentry_reaction[0].parse_results))

				# print(type(subentry_reaction[0]))
				# print(str(subentry_reaction[0])+'jjjjjjjjj')

				if multiple_product_subentries:
					if subentry_reaction[0].residual == None:
						label_string = author_name+' ('+year+') ['+str(subentry_reaction[0].targ)+'('+self.exfor_reaction.replace('*','X').lower()+')'+str(subentry_reaction[0]).split('Unspecified+')[1].strip(')')+']'
					else:
						label_string = author_name+' ('+year+') ['+str(subentry_reaction[0].targ)+'('+self.exfor_reaction.replace('*','X').lower()+')'+str(subentry_reaction[0].residual)+']'
				else:
					label_string = author_name+' ('+year+')'

				if unc_xs_col == -1 and unc_energy_col == -1:
					# 2-column data, energy and xs...
					# print('plotting 2-column')
					# print(plot_Dict[index][2][:,0])
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar,  ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)
					# 
				elif unc_energy_col == -1:
					# 3-column data, energy, xs, and xs uncertainty...
					# print('plotting 3-column')
					# print(plot_Dict[index][2][:,0])
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar,  yerr=plot_data[:,unc_xs_col]*xs_unit_scalar, ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)
				else:
				# 	# 4-column data, energy, xs,  xs uncertainty, and energy uncertainty...
					# print('plotting 4-column')
				# 	# print(plot_Dict[index][2][:,0])
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar, xerr=plot_data[:,unc_energy_col]*energy_unit_scalar, yerr=plot_data[:,unc_xs_col]*xs_unit_scalar, ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)

				
				# print(plot_Dict[index][2].shape[1])
				# if plot_Dict[index][2].shape[1] == 4:
				# 	# print('plotting scatter')
				# 	# print(plot_Dict[index][2][:,0])
				# 	plt.errorbar(plot_Dict[index][2][:,0],1E3*plot_Dict[index][2][:,1], xerr=plot_Dict[index][2][:,2], yerr=1E3*plot_Dict[index][2][:,3], ls='none', capsize=3, label=plot_Dict[index][0]+' ('+plot_Dict[index][1]+')', marker='o', markersize=3, linewidth=1)
				# elif plot_Dict[index][2].shape[1] == 3:
				# 	# print(plot_Dict[index][2][0])
				# 	# print(print(plot_Dict[index][2][:,1]))
				# 	plt.errorbar(plot_Dict[index][2][:,0],1E3*plot_Dict[index][2][:,1], yerr=1E3*plot_Dict[index][2][:,2], ls='none', capsize=3, label=plot_Dict[index][0]+' ('+plot_Dict[index][1]+')', marker='o', markersize='3', linewidth=1)
				# elif plot_Dict[index][2].shape[1] >= 5:
				# 	print('WARNING: Plotting',str(plot_Dict[index][2].shape[1])+'-column EXFOR data retrieved for subentry', plot_Dict[index][3]+', please make sure data look reasonable - column formatting is inconsistent for >4 columns.')
				# 	# print(plot_Dict[index][2][0])
				# 	# print(print(plot_Dict[index][2][:,1]))
				# 	plt.errorbar(plot_Dict[index][2][:,4],plot_Dict[index][2][:,5], xerr=plot_Dict[index][2][:,0], yerr=plot_Dict[index][2][:,6], ls='none', capsize=3, label=plot_Dict[index][0]+' ('+plot_Dict[index][1]+')', marker='o', markersize='3', linewidth=1)
				k=k+1



			# target_url_209po = 'https://tendl.web.psi.ch/tendl_2021/deuteron_file/Bi/Bi209/tables/residual/rp084209.tot'
			# target_url_208po = 'https://tendl.web.psi.ch/tendl_2021/deuteron_file/Bi/Bi209/tables/residual/rp084208.tot'
			# tendl_data = np.genfromtxt(urlopen(target_url_209po), delimiter=" ")
			# tendl_data_208 = np.genfromtxt(urlopen(target_url_208po), delimiter=" ")
			# Not using urlopen() is faster, but saves a local copy of the downloaded file - might be useful, I find it annoying!
			# tendl_data = np.genfromtxt(urlopen(target_url_209po), delimiter=" ")



			



			if self.plot_tendl:
				element = Element(self.target_element)
				abd = element.abundances
				# print(element.isotopes)
				# print(type(abd))
				abundances = abd.loc[:,'abundance'].to_numpy()
				isotopes = abd.loc[:,'isotope'].to_numpy()
				# print(abundances)
				# print(isotopes)

				# print(len(isotopes))

				if len(isotopes) == 0:
					# Likely a radioactive target
					isotopes = self.exfor_target
					abundances = 100.0
					self.enriched = True

				print(self.exfor_product)

				product_tendl=self.exfor_product.split('-')[1]+self.exfor_product.split('-')[0]+'g'
				if self.enriched:
					rx = Reaction(self.exfor_target.split('-')[1]+self.exfor_target.split('-')[0]+'(p,x)'+product_tendl)
					tendl_xs = rx.xs
				else:
					rx = Reaction(isotopes[0]+'(p,x)'+product_tendl)
					tendl_xs = np.zeros(len(rx.eng))
					for (itp, abund) in zip(isotopes,abundances):
						tendl_xs = tendl_xs + (Reaction(itp+'(p,x)'+product_tendl).xs * (abund/100))
				# print(rx.xs)
				# print(rx.eng)
				# print(tendl_xs)

				plt.plot(rx.eng, tendl_xs, color="k", label='TENDL')

			print('Plotting '+str(len(plot_Dict))+' datasets found in EXFOR for '+self.exfor_target+'('+self.exfor_reaction.replace('*','X')+')'+self.exfor_product)
			plt.legend()
			plt.xlabel('Incident Energy (MeV)')
			plt.ylabel('Cross Section (mb)')
			# plt.xlim(right=50)
			# plt.xlim([0,50])
			# plt.ylim(top=1000)
			plt.ylim(bottom=0)
			plt.title(self.exfor_target.capitalize()+'('+self.exfor_reaction.lower().replace('*','x')+')'+self.exfor_product.capitalize())
			# plt.show()
			plt.grid(which='major', axis='both', color='w')

			ax = plt.gca()
			ax.set_facecolor('#e5ecf6')

			# # Set up second y-axis for plotting isotopic ratios
			# ax2=ax.twinx()
			# # ax2.semilogy(tendl_data_208[:,0], tendl_data[:,1]/(tendl_data_208[:,1]), label='Isotopic Ratio', color='g')
			# # ax2.plot(tendl_data_208[:,0], tendl_data[:,1]/(tendl_data_208[:,1]), label='Isotopic Ratio', color='g')
			# ax2.plot(tendl_data_208[:,0], tendl_data[:,1]/(tendl_data[:,1]+tendl_data_208[:,1]), label='Isotopic Ratio', color='g')
			# # ax2.set_ylabel('Predicted 209Po:208Po Yield (Mass Ratio)', color='g')
			# ax2.set_ylabel('Predicted Fraction $^{209Po}$/$_{209Po+208Po}$ Yield (Mass Ratio)', color='g')
			# ax2.set_ylim([0,1.2])
			# ax2.tick_params(axis='y', labelcolor='g')
			# # plt.legend()
			# plt.savefig('165Er_XS.png', dpi=400)  
			plt.show()
		else:
			print('No matching datasets found!')


	