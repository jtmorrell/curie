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
		self.plot_Dict = {}
		self.multiple_product_subentries = False
		self.enriched = False

		if '*' in self.name:
			self.library = Library('exfor')

		if library.lower()=='best':
			if self.incident=='n':
				for lb in ['irdff','endf','iaea','tendl','tendl_n_rp']:
					self.library = Library(lb)
					if '*' in self.name:
						self.library = Library('exfor')
					elif lb=='tendl_n_rp':
						self._check(True)
					elif self._check():
						break
			elif self.incident in ['p','d']:
				for lb in ['iaea','tendl_'+self.incident+'_rp']:
					self.library = Library(lb)
					if '*' in self.name:
						self.library = Library('exfor')
					elif lb=='tendl_d_rp':
						self._check(True)
					elif self._check():
						break
			else:
				try:
					self.library = Library('iaea')
					self._check(True)
				except ValueError:
					self.library = Library('exfor')
		else:
			self.library = Library(library)
			self._check(True)

		if 'a' in self.name:
			self.library = Library('exfor')
		elif '*' not in self.name:
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
		else:
			self.library = Library('exfor')

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
		"""Reaction code conversion

		Setter method to convert Curie-formatted reaction codes into EXFOR-formatted 
		codes, used in the query(), search_exfor(), plot_exfor() methods. This method
		is unlikely to be directly called in general use cases

		Returns
		-------
		self.exfor_target : str
			The target nucleus (or nuclei), in EXFOR notation.

		self.exfor_reaction : str
			The reaction code, in EXFOR notation.

		self.exfor_product : str
			The product nucleus (or nuclei), in EXFOR notation.

		Examples
		--------
		>>> self.exfor_target, self.exfor_reaction, self.exfor_product = self.curie_to_exfor()

		"""
		# Parse curie reaction name components
		parsed_target = self.name.split('(')[0].strip('g')
		reaction_code = self.name.split('(')[1].split(')')[0].strip('g')
		parsed_product = self.name.split('(')[1].split(')')[1].strip('g')

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
		else:
			substrings = re.split('(\D+)',parsed_target)
			self.exfor_target = (substrings[1]+'-'+substrings[0]).upper()


		# Look for ZZZ-* (any isotope) products
		if '*' in parsed_product:
			# print('found!')
			self.exfor_product = parsed_product.replace('*','9999')
		else:
			substrings = re.split('(\D+)',parsed_product)
			self.exfor_product = (substrings[1]+'-'+substrings[0]).upper()


		self.exfor_reaction = reaction_code.upper().replace('X','*')

		return self.exfor_target, self.exfor_reaction, self.exfor_product


	def search_exfor(self, plot_results=False, plot_tendl=False, show_legend=False, xlim=[0,None], ylim=[0,None], verbose=False):
		"""Search the EXFOR library for reaction data

		Front-end method to search for EXFOR data for the reaction (or reactions) defined in
		ci.Reaction(). Unlike reaction data retrieved from other Curie data libraries, this search
		supports the use of natural abundance targets and the use of wildcards for both target, 
		reaction channel, and product. Additional optional parameters control the plotting and 
		level of verbose output. This provides an API-like access for EXFOR data, but users may 
		want to compare against queries through the EXFOR web portal, due to the fact that EXFOR 
		compilations may be entered inconsistently, particularly for older data sets.

		The first time that Reaction.search_exfor() is called, Curie will download the most recent 
		copy of the EXFOR database. The x4i3-tools Python package provides tools to manually update 
		your local EXFOR database, but native upgrade support will be added in future updates.

		Currently, supported targets include individual isotopes (as in the rest of Curie) (e.g., 
		'226RA'), natural abundance targets (e.g., '0FE', '0PB'), and wildcards (e.g., '*LA', '*V'), 
		which which includes both A=0 for the specified element, as well as all of its stable 
		isotopes. The target element must still be specified, but support for wildcard elements (e.g., 
		'*(p,x)56CO') will be added in future updates. 

		Similarly, reaction channels may be specified as normal in Curie (e.g., '(p,x)', '(p,2n)', 
		'd,a4n'), or using wildcards for exit channels (e.g., '(p,x)', '(p,*)', where 'x' and '*' 
		serve interchangeably), though the incident particle must still be specified.

		Currently, supported products include individual isotopes (as in the rest of Curie) (e.g., 
		'225RA'), and wildcards (e.g., '*CO', '*CE', '*TL'), which will retrieve results for all  
		isotopes of the specified product (or wildcard target). This also includes wildcard elements 
		(e.g., '56FE(p,x)*'). When plotting results for a wildcard product, the plot legend will provide  
		extra metadata to specify the reaction channel for each dataset.

		Combinations of any of the above supported wildcards may be used together and interchangeably, 
		offering data retrieval power comparable to the EXFOR web portal, but with greater ease for
		programmatic use.
		

		Parameters
		----------
		plot_results : bool, optional
			When 'True', this parameter is used to plot results retrieved through Reaction.search_exfor(). 
			Default 'False'.

		plot_tendl : bool, optional
			When 'True', this parameter is used to plot TENDL results for comparison with retrieved EXFOR 
			data. Only works for non-wildcard products, and for incident p,d,n currently, as other 
			incident particles are not available in Library() for TENDL currently. Default 'False'.

		show_legend : bool, optional
			When 'True', the plots generated in Reaction.plot_exfor() will have the legend displayed. 
			Default 'False'.

		xlim : arraylike, optional
			An array that sets the x-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(xlim) == 1, this sets the lower bound only. If len(xlim) == 2, then 
			xlim sets both upper and lower bounds as xlim = [lower, upper].

		ylim : arraylike, optional
			An array that sets the y-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(ylim) == 1, this sets the lower bound only. If len(ylim) == 2, then 
			ylim sets both upper and lower bounds as ylim = [lower, upper].

		verbose : bool, optional
			This enables verbose output to the terminal for results retrieved from EXFOR. Useful for 
			examining the raw data or for viewing metadata.		


		Examples
		--------
		>>> rx = ci.Reaction('*LA(p,x)139CEg')
		>>> rx.search_exfor()
		>>> 
		>>> rx = ci.Reaction('27AL(n,x)24NAg')
		>>> rx.search_exfor(plot_results=True,plot_tendl=True, show_legend=True, xlim=[None,None], verbose=False)
		>>>
		>>> rx = ci.Reaction('226RA(n,x)225RA')
		>>> rx = ci.Reaction('226RA(n,x)224RA')
		>>> rx = ci.Reaction('226RA(n,x)*RA')
		>>> rx = ci.Reaction('226RA(n,x)*')
		>>>
		>>>
		>>> rx = ci.Reaction('27AL(n,x)24NA')
		>>> rx = ci.Reaction('*AL(n,x)24NA')
		>>> rx = ci.Reaction('27AL(n,x)*NA')
		>>> rx = ci.Reaction('*AL(n,x)*')
		>>>
		>>>
		>>> rx = ci.Reaction('103RH(p,x)103PD')
		>>> rx = ci.Reaction('*RH(p,x)103PD')
		>>> rx = ci.Reaction('103RH(p,x)*PD')
		>>> rx = ci.Reaction('*RH(p,x)*PD')
		>>> rx = ci.Reaction('*RH(p,x)*')    
		>>>
		>>>
		>>> rx = ci.Reaction('107AG(p,x)107CD')
		>>> rx = ci.Reaction('*AG(p,x)107CD')
		>>> rx = ci.Reaction('107AG(p,x)*CD')
		>>> rx = ci.Reaction('*AG(p,x)*AG')
		>>> rx = ci.Reaction('*AG(p,x)*')
		>>>
		>>>
		>>> rx = ci.Reaction('*AS(p,x)*AS')
		>>> rx = ci.Reaction('*AS(p,x)*GE')
		>>> rx = ci.Reaction('*AS(p,x)*')
		>>> rx = ci.Reaction('*AS(p,x)*SE')
		>>>
		>>>
		>>> rx = ci.Reaction('0LA(p,x)*')
		>>> rx = ci.Reaction('0LA(p,x)*CE')
		>>> rx = ci.Reaction('*LA(p,x)*CE')
		>>> rx = ci.Reaction('*LA(p,x)139CE')
		>>> rx = ci.Reaction('*LA(p,x)134CE')
		>>> rx = ci.Reaction('*LA(p,x)135CE')
		>>> rx = ci.Reaction('*LA(p,x)*')
		>>>
		>>>
		>>> rx = ci.Reaction('*BI(a,x)*')
		>>> rx = ci.Reaction('*OS(a,x)*')
		>>> rx = ci.Reaction('*OS(a,x)*PT')
		>>> rx = ci.Reaction('*OS(a,x)*IR')
		>>>
		>>>
		>>> rx = ci.Reaction('*FE(d,x)*CO')
		>>> rx = ci.Reaction('0FE(d,x)*CO')
		>>> rx = ci.Reaction('*FE(d,x)56CO')
		>>> rx = ci.Reaction('*FE(d,x)*')
		>>> 
		>>> rx.search_exfor(verbose=True)


		"""

		# Check for *-targets (aka A=9999)
		self.multiple_product_subentries = False


		if '9999' not in self.exfor_product:
			# Loop over all stable isotopes of element, along with A=0 (natural abundance)
			self.plot_tendl = plot_tendl

		if '9999' in self.exfor_target:
			# Loop over all stable isotopes of element, along with A=0 (natural abundance)
			self.exfor_target = self.exfor_target.strip('9999')
			# Avoid a spaghetti plot...
			if '9999' not in self.exfor_product:
				# Loop over all stable isotopes of element, along with A=0 (natural abundance)
				self.plot_tendl = plot_tendl
			else:
				self.plot_tendl = False
			self.multiple_product_subentries = True
			element = Element(self.exfor_target)
			abd = element.abundances
			isotopes = abd.loc[:,'isotope'].to_numpy()
			isotopes = np.append(isotopes, '0'+self.exfor_target)

			# Loop over all stable and nat- isotopes, and query EXFOR for each
			for itp in isotopes:
				substrings = re.split('(\D+)',itp)
				self.exfor_target = (substrings[1]+'-'+substrings[0]).upper()
				self.query(plot_results, self.plot_tendl, show_legend, xlim, ylim, verbose)

		else:
			# Run an EXFOR query as normal
			self.query(plot_results, self.plot_tendl, show_legend, xlim, ylim, verbose)
		

	def query(self, plot_results=False, plot_tendl=False, show_legend=False, xlim=[0,None], ylim=[0,None], verbose=False):
		"""EXFOR retrieval method

		Back-end method that takes the list of element(s) and/or isotope(s) from Reaction.search_exfor(), 
		passes along any plotting options, and actually retrieves the data.  This does all of the heavy 
		lifting, but users will likely never need to call this method.

		Parameters
		----------
		plot_results : bool, optional
			When 'True', this parameter is used to plot results retrieved through Reaction.search_exfor(). 
			Default 'False'.

		plot_tendl : bool, optional
			When 'True', this parameter is used to plot TENDL results for comparison with retrieved EXFOR 
			data. Only works for non-wildcard products, and for incident p,d,n currently, as other 
			incident particles are not available in Library() for TENDL currently. Default 'False'.

		show_legend : bool, optional
			When 'True', the plots generated in Reaction.plot_exfor() will have the legend displayed. 
			Default 'False'.

		xlim : arraylike, optional
			An array that sets the x-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(xlim) == 1, this sets the lower bound only. If len(xlim) == 2, then 
			xlim sets both upper and lower bounds as xlim = [lower, upper].

		ylim : arraylike, optional
			An array that sets the y-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(ylim) == 1, this sets the lower bound only. If len(ylim) == 2, then 
			ylim sets both upper and lower bounds as ylim = [lower, upper].

		verbose : bool, optional
			This enables verbose output to the terminal for results retrieved from EXFOR. Useful for 
			examining the raw data or for viewing metadata.		

		Examples
		--------
		>>> self.query(plot_results, self.plot_tendl, show_legend, xlim, ylim, verbose)

		"""

		# Load the EXFOR database from local site-packages
		db = exfor_manager.X4DBManagerDefault()

		# Extract name of specified target element for later string formatting
		self.target_element = self.exfor_target.split('-')[0].capitalize()

		# Local bool - used for specifying '*' as a product for full wild card mode
		is_true_wildcard_product = False


		# Check for *-products (aka A=9999)
		if '9999' in self.exfor_product:
			# title_product = product.replace('9999','*')
			self.exfor_product = self.exfor_product.strip('9999')
			if self.exfor_product == '':
				print('SUPER WILD MODE - querying ALL products')
				is_true_wildcard_product = True
			# Avoid a spaghetti plot...
			self.plot_tendl = False
			self.multiple_product_subentries = True
		
		# Check for *-targets (aka A=9999)
		if '9999' in self.exfor_target:
			self.exfor_target = self.exfor_target.strip('9999')
			# Avoid a spaghetti plot...
			self.plot_tendl = False
			self.multiple_product_subentries = True
		else:
			# Detect if target material is natural abundance
			if str(self.exfor_target.split('-')[1]) != str(0):
				self.enriched = True
			else:
				self.enriched = False



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

			datasets = entry.getDataSets()

			# Find columns for relevant data
			energy_col = -1
			unc_energy_col = -1
			xs_col = -1
			unc_xs_col = -1
			xs_unit_scalar = 1
			energy_unit_scalar = 1
			xs_unc_unit_scalar = 1
			energy_unc_unit_scalar = 1

			# Make sure only one subentry is retrieved per entry!
			num_of_sub_subentries = len(list(datasets.keys()))

			subentry_list = []

			if num_of_sub_subentries == 0:
				continue
			elif num_of_sub_subentries == 1:
				# We're all good...
				i = next(iter(datasets.values()))
				# Only select subentries leading to the specified product
				if self.exfor_product in str(i.reaction) and "NON" not in str(i.reaction) and "TTY" not in i.reaction[0].quantity and "RECOM" not in i.reaction[0].quantity and "EVAL" not in i.reaction[0].quantity:
					subentry_list.append(i)
				else:
					continue

			else:
				# Poorly-formatted EXFOR - multiple subentries for one entry
				if verbose:
					print('Number of datasets found in entry', next(iter(datasets))[1][0:5], ': ', num_of_sub_subentries)

				for i in datasets.values():
					# Only select subentries leading to the specified product
					if self.exfor_product in str(i.reaction) and "NON" not in i.reaction[0].quantity and "TTY" not in i.reaction[0].quantity and "RECOM" not in i.reaction[0].quantity and "EVAL" not in i.reaction[0].quantity:
						subentry_list.append(i)


			

			# Loop to next key if we didnt find any matches
			if len(subentry_list) == 0:
				continue


			# Pull metadata and stash into dictionary
			for subentry in subentry_list:
				xs_col = -1
				unc_xs_col = -1
				unc_energy_col = -1
				energy_col = -1

				# Verbose output of ALL data for each retrieved subentry
				if  verbose:
					print(subentry)
					print(subentry.subent)
					print(str(subentry.reaction))

				# Make sure data is ACTUALLY XS data - DERIV is borderline, but likely okay
				if "RECOM" in str(subentry.reaction):
					if  verbose:
						print('RECOM data found in subentry',subentry.subent,', skipping...')
					continue
				elif "TTY" in str(subentry.reaction):
					if  verbose:
						print('TTY data found in subentry',subentry.subent,', skipping...')
					continue
				elif "SFC" in str(subentry.reaction):
					if  verbose:
						print('SFC data found in subentry',subentry.subent,', skipping...')
					continue
				elif "/" in str(subentry.reaction):
					if  verbose:
						print('Ratio data found in subentry',subentry.subent,', skipping...')
					continue
				elif "PAR" in str(subentry.reaction):
					if  verbose:
						print('Partial XS data found in subentry',subentry.subent,', skipping...')
					continue
				elif "REL" in str(subentry.reaction):
					if  verbose:
						print('REL data found in subentry',subentry.subent,', skipping...')
					continue
				# elif "DERIV" in str(subentry.reaction):
				# 	if  verbose:
				# 		print('DERIV data found in subentry',subentry.subent,', skipping...')
				# 	continue
				elif "RAW" in str(subentry.reaction):
					if  verbose:
						print('RAW data found in subentry',subentry.subent,', skipping...')
					continue

				# Find + extract column data - EXFOR doesn't use a uniform ordering
				for entry in subentry.labels:
					# Find what column the energy data are located in
					if entry in ("Energy", "EN"): 
						energy_col = subentry.labels.index(entry)
					# Find what column the xs data are located in
					if entry.casefold() == "Data".casefold(): 
						xs_col = subentry.labels.index(entry)
					if entry in ("d(Data)", "DATA-ERR", "ERR-T"): 
						if subentry.units[subentry.labels.index(entry)].casefold() not in ("NO-DIM".casefold()):
							unc_xs_col = subentry.labels.index(entry)
						else:
							continue
					if entry in ("d(Energy)", "EN-ERR"): 
						unc_energy_col = subentry.labels.index(entry)
					elif entry in ("EN-RSL", "EN-RSL-FW"): 
						unc_energy_col = subentry.labels.index(entry)


				if energy_col == -1: 
					if  verbose:
						print('Center-of-mass data found in subentry',subentry.subent,', skipping...')
					continue


				# ...grab the energy units
				if subentry.units[energy_col] in ("MEV", "MeV"):
					energy_unit_scalar = 1
				elif subentry.units[energy_col] in ("KEV", "keV"):
					energy_unit_scalar = 1E-3
				elif subentry.units[energy_col] in ("EV", "eV"):
					energy_unit_scalar = 1E-6
				else:
					print("No energy units found for entry", subentry.subent)
					energy_unit_scalar = -1
					continue
				# same for energy uncertainty
				if subentry.units[unc_energy_col] in ("MEV", "MeV"):
					energy_unc_unit_scalar = 1
				elif subentry.units[unc_energy_col] in ("KEV", "keV"):
					energy_unc_unit_scalar = 1E-3
				else:
					energy_unc_unit_scalar = 0

				# ...grab the xs units
				if subentry.units[xs_col].casefold() == "mb".casefold():
					xs_unit_scalar = 1
				elif subentry.units[xs_col].casefold() in ("barns".casefold(), "barn".casefold(), "B".casefold()):
					xs_unit_scalar = 1E3
				elif subentry.units[xs_col].casefold() in ("ubarns".casefold(), "ub".casefold(), "MICRO-B".casefold()):
					xs_unit_scalar = 1E-3
				elif subentry.units[xs_col].casefold() in ("PER-CENT".casefold()):
					xs_unit_scalar = np.array(subentry.data, dtype=float)[:,xs_col]
				else:
					print("No XS units found for entry", subentry.subent)
				# same for xs uncertainty
				if subentry.units[unc_xs_col].casefold() == "mb".casefold():
					xs_unc_unit_scalar = 1
				elif subentry.units[unc_xs_col].casefold() in ("barns".casefold(), "barn".casefold(), "B".casefold()):
					xs_unc_unit_scalar = 1E3
				elif subentry.units[unc_xs_col].casefold() in ("ubarns".casefold(), "ub".casefold(), "MICRO-B".casefold()):
					xs_unc_unit_scalar = 1E-3
				elif subentry.units[unc_xs_col].casefold() in ("PER-CENT".casefold()):
					xs_unc_unit_scalar = np.array(subentry.data, dtype=float)[:,xs_col]*0.01*xs_unit_scalar
				else:
					print("No XS uncertainty units found for entry", subentry.subent)
					xs_unc_unit_scalar = -1

				if xs_col == -1:
					print('No XS able to be extracted for entry', subentry.subent)
					continue

				# Avoid later crashes if a column has a header, but no data
				try:
					np.array(subentry.data, dtype=float)[:,unc_xs_col]
				except IndexError:
					unc_xs_col = -1


				# Pretty format metedata
				author_name = subentry.author[0].split('.',-1)[-1]
				year = subentry.year

				# Avoid non-residual product cross sections
				if 'Mass' not in str(subentry.reaction[0]) and ')+()' not in str(subentry.reaction[0]) and 'Fission' not in str(subentry.reaction[0]) and 'Total' not in str(subentry.reaction[0]) and  'Elastic' not in str(subentry.reaction[0]): 
					try:
						if subentry.reaction[0].residual == None:
							if 'Unspecified' not in str(subentry.reaction[0]):
								continue
							else:
								residual = str(subentry.reaction[0]).split('Unspecified+')[1].strip(')')
						else:
							residual = str(subentry.reaction[0].residual)

						# Perform final checks on if the target element also matches
						if self.exfor_product == '':
							is_true_wildcard_product = True
						if is_true_wildcard_product:
							# stash into dictionary
							plot_Dict[author_name+year+subentry.subent+residual] = (author_name, # 0
													  year, # 1
													  np.array(subentry.data, dtype=float),  # 2
													  subentry.subent, # 3
													  energy_unit_scalar, # 4
													  xs_unit_scalar, # 5
													  energy_col, # 6
													  xs_col, # 7
													  unc_energy_col, # 8 
													  unc_xs_col,  # 9
													  subentry.reaction,  # 10
													  residual,  #11
													  energy_unc_unit_scalar,   #12
												      xs_unc_unit_scalar)   #13
						else:
							# make sure we don't falsely match if the target element and product are the same
							wildcard_substrings = re.split('(\D+)',self.exfor_product)
							if str(wildcard_substrings[1]).upper().strip('-') in str(residual).upper():
								plot_Dict[author_name+year+subentry.subent+residual] = (author_name, # 0
														  year, # 1
														  np.array(subentry.data, dtype=float),  # 2
														  subentry.subent, # 3
														  energy_unit_scalar, # 4
														  xs_unit_scalar, # 5
														  energy_col, # 6
														  xs_col, # 7
														  unc_energy_col, # 8 
														  unc_xs_col,  # 9
														  subentry.reaction,  # 10
														  residual,  #11
														  energy_unc_unit_scalar,   #12
													      xs_unc_unit_scalar)   #13
					except AttributeError:
						continue

		# Update selected data in dictionary, caching it for later plotting
		self.plot_Dict = plot_Dict

		# Adds divider liens to break up the LONG verbose output
		if verbose:
			print('---------------------------')


		# If specified by user, go ahead and plot results
		if plot_results:
			self.plot_exfor(self.plot_tendl,show_legend, xlim=xlim, ylim=ylim)
		


	def plot_exfor(self, plot_tendl=False, show_legend=False, xlim=[None,None], ylim=[0,None]):
		"""Plot the EXFOR data

		Plot the EXFOR data retrieved by Reaction.search_exfor().  This method can called as 
		part of Reaction.search_exfor(), or can be called separately for plotting results which have 
		already been retrieved.

		Parameters
		----------
		plot_tendl : bool, optional
			When 'True', this parameter is used to plot TENDL results for comparison with retrieved EXFOR 
			data. Only works for non-wildcard products, and for incident p,d,n currently, as other 
			incident particles are not available in Library() for TENDL currently. Default 'False'.

		show_legend : bool, optional
			When 'True', the plots generated in Reaction.plot_exfor() will have the legend displayed. 
			Default 'False'.

		xlim : arraylike, optional
			An array that sets the x-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(xlim) == 1, this sets the lower bound only. If len(xlim) == 2, then 
			xlim sets both upper and lower bounds as xlim = [lower, upper].

		ylim : arraylike, optional
			An array that sets the y-axis plot limits in Reaction.plot_exfor(), implemented in 
			matplotlib.  If len(ylim) == 1, this sets the lower bound only. If len(ylim) == 2, then 
			ylim sets both upper and lower bounds as ylim = [lower, upper].



		Examples
		--------
		>>> rx = ci.Reaction('*LA(p,x)*CE')
		>>> rx.plot_exfor(plot_tendl=True, show_legend=True, xlim=[0,100])
		>>> rx.plot_exfor(plot_tendl=True, show_legend=True, xlim=[0,850], ylim=[0, None])

		"""

		# Update local variables, if called as part of a loop
		plot_Dict = self.plot_Dict
		# Make sure we dont have a A=* or exfor_product = '*' by accident
		if len(self.exfor_product.split('-')) ==1:
			plot_tendl = False
		self.plot_tendl = plot_tendl
	

		# Skip plotting if no results have been retrieved 
		if len(plot_Dict) != 0:
			# Plot results
			fig, ax = plt.subplots(layout='constrained')

			k=0
			for index in plot_Dict:
				# Need to set up list of marker sizes to iterate over with k
				# Use local variables from dictionary value
				author_name, year , plot_data , subent , energy_unit_scalar , xs_unit_scalar , energy_col , xs_col , unc_energy_col , unc_xs_col, subentry_reaction, residual, energy_unc_unit_scalar, xs_unc_unit_scalar = plot_Dict[index]
			
				# Properly define the label string, based on type of entry
				if self.multiple_product_subentries:
					if subentry_reaction[0].residual == None:
						label_string = author_name+' ('+year+') ['+str(subentry_reaction[0].targ)+'('+self.exfor_reaction.replace('*','X').lower()+')'+residual+']'
					else:
						label_string = author_name+' ('+year+') ['+str(subentry_reaction[0].targ)+'('+str(subentry_reaction).split('(')[2].split(')')[0].replace('*','X').lower()+')'+residual+']'
				else:
					label_string = author_name+' ('+year+')'

				# Determine which type of dataset is being plotted
				if unc_xs_col == -1 and unc_energy_col == -1:
					# 2-column data, energy and xs...
					# print('plotting 2-column')
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar,  ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)
					# 
				elif unc_xs_col != -1 and unc_energy_col == -1:
					# 3-column data, energy, xs, and xs uncertainty...
					# print('plotting 3-column')
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar,  yerr=plot_data[:,unc_xs_col]*xs_unc_unit_scalar, ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)
				elif unc_xs_col == -1 and unc_energy_col != -1:
					# 3-column data, energy, xs, and energy uncertainty...
					# print('plotting 3-column')
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar,  xerr=plot_data[:,unc_energy_col]*energy_unc_unit_scalar, ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)
				elif unc_xs_col != -1 and unc_energy_col != -1:
					# 4-column data, energy, xs,  xs uncertainty, and energy uncertainty...
					# print('plotting 4-column')
					plt.errorbar(plot_data[:,energy_col]*energy_unit_scalar,plot_data[:,xs_col]*xs_unit_scalar, xerr=plot_data[:,unc_energy_col]*energy_unc_unit_scalar, yerr=plot_data[:,unc_xs_col]*xs_unc_unit_scalar, ls='none', capsize=3, label=label_string, marker='o', markersize=3, linewidth=1)

				k=k+1

			# Make sure to not plot TENDL for unsupported incident particles
			if self.incident not in ['p','d','n']:
				print('Plotting TENDL data currently limited to p,d,n as incident particles, sorry!')
				self.plot_tendl = False
			

			if self.plot_tendl:
				element = Element(self.target_element)
				abd = element.abundances
				abundances = abd.loc[:,'abundance'].to_numpy()
				isotopes = abd.loc[:,'isotope'].to_numpy()


				if len(isotopes) == 0:
					# Likely a radioactive target
					isotopes = self.exfor_target
					abundances = 100.0
					self.enriched = True

				# Make sure we don't have an enriched target
				nearly_monoisotopic_elements = ['H', 'N', 'LA', 'HE', 'O', 'TA', 'C', 'V']
				if self.target_element.upper() in nearly_monoisotopic_elements:
					self.enriched = False

				# Format the EXFOR reaction string into TENDL notation
				product_tendl=self.exfor_product.split('-')[1]+self.exfor_product.split('-')[0]+'g'
				if self.enriched:
					rx = Reaction(self.exfor_target.split('-')[1]+self.exfor_target.split('-')[0]+'('+self.incident+',x)'+product_tendl)
					tendl_xs = rx.xs
				else:
					rx = Reaction(isotopes[0]+'('+self.incident+',x)'+product_tendl)
					tendl_xs = np.zeros(len(rx.eng))
					for (itp, abund) in zip(isotopes,abundances):
						try:
							tendl_xs = tendl_xs + (Reaction(itp+'('+self.incident+',x)'+product_tendl).xs * (abund/100))
						except ValueError:
							continue
				# Plot TENDL
				plt.plot(rx.eng, tendl_xs, color="k", label='TENDL')

			print('Plotting '+str(len(plot_Dict))+' datasets found in EXFOR for '+self.exfor_target+'('+self.exfor_reaction.replace('*','X')+')'+self.exfor_product)

			# Set up axes and figure for pretty figure style
			ax = plt.gca()
			if show_legend:
				legend = fig.legend(loc='outside right upper')
				fig_width, fig_height = plt.gcf().get_size_inches()
				fig.set_figwidth(plt.rcParams['figure.figsize'][0]*(1+(legend.get_window_extent().width/(100*fig_width))))

			plt.xlabel('Incident Energy (MeV)')
			plt.ylabel('Cross Section (mb)')
			plt.xlim(xlim)
			plt.ylim(ylim)
			plt.title(self.exfor_target.capitalize()+'('+self.exfor_reaction.lower().replace('*','x')+')'+self.exfor_product.capitalize())
			plt.grid(which='major', axis='both', color='w')

			ax.set_facecolor('#e5ecf6')
			plt.show()
		else:
			print('No matching datasets found for '+self.exfor_target+'('+self.exfor_reaction.replace('*','X')+')'+self.exfor_product)

		print('---------------------------')


	