import curie as ci
import numpy as np
import pandas as pd
# import warnings
# import itertools
import matplotlib.pyplot as plt
from scipy import interpolate
from packaging import version
# warnings.simplefilter(action='ignore', category=FutureWarning)

class Yield(object):
	"""Calculate activation yields

	The Yield class performs calculations of yields for one or more targets, using the 
	Stack class to perform transport calculations.
	
	Parameters
	----------
	stack_file :  str
		Definition of the foils in the stack, inherited from the Stack class.  The 'compound' for each foil in
		the stack must be given, and the 'areal_density' or some combination of parameters
		that allow the areal density to be calculated must also be given.  Foils must
		also be given a 'name' if they are to be filtered by the .saveas(), .summarize(),
		and .plot() methods.  By default, foils without 'name' are not included by these
		methods.

		There are three acceptable formats for `stack`, inherited from the Stack class. 
		However, preferred is a str, which is a path to a file in either .csv,
		.json or .db format, where the headers of the file contain the correct information.
		Note that the .json file must follow the 'records' format (see pandas docs).  If a .db
		file, it must have a table named 'stack'. This route is preferred, as it allows users
		to simply perform the yield calculations on an existing stack file output.



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


	# beam_current = 1e3               # nA
	# irradiation_length = 1*3600        # s  - half-lives are being pulled in in units of s
	# cooling_length = 24.0               # h  - time after EoB to report activities
	# lower_halflife_threshold = 60    # s    - Remove any products which are unreasonably short
	# show_plots = False               # Show plots of the compound-calculated cross sections
	# save_csv = True                  # Export results to csv - MASSIVELY reduces computational time for subsequent runs
	# summary_masses = True            # In addition to activity calculations, calculate masses of all reaction products
	# particle = 'd'					 # Options are 'p',  'd'   (for now)
	# # E0 = 29.1 						 # MeV  -  incident particle energy
	# # E0 = 15.11 						 # MeV  -  incident particle energy
	# E0 = 25.0 						 # MeV  -  incident particle energy
	# compound_cross_sections_csv = 'compound_cross_sections.csv'    # Output file name for the csv containing XS data for compounds.  Generation of this takes 
	#                                                                # much longer than doing the activity/yield calculations, so exporting and loading this files
	#                                                                # speeds up runs, as long as this file contains data for all compounds in your stack
	# n_largest_activities = 20         # Return the n largest (e.g., the largest 3) activities in each foil for summary 
	# activity_units = 'uCi'           # Specifies activity unit output - options are 'Bq', 'kBq', 'MBq', 'GBq', 'uCi', 'mCi', 'Ci'
	# mass_units = 'ug'                # Specifies mass unit output - options are 'fg', 'pg', 'ng', 'ug', 'mg'


	# # use_enriched_targets = False      # now auto-detected, based on if enriched_targets is empty or non-empty
	# # enriched_targets = ['186W']  
	# enriched_targets = {}  # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'W': {'180W': 0.0012, '182W': 0.2650, '183W': 0.1431, '184W': 0.3064, '186W': 0.2843}}  # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'W': {'186W': 1.0}} # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'Ti': {'186W': 1.0}} # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'Pa': {'231PA': 1.0}} # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'Ni': {'60NI': 1.0}} # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0
	# # enriched_targets = {'Yb': {'176YB': 1.0}} # Optional, overrides natural abundance elements with dictionary of enriched targets.  Will get renormalized if enrichments do not sum to 1.0

	"""

	def __init__(self, stack_file, particle='p', E0=30.0, beam_current=1e3, irradiation_length=3600.0, **kwargs):
		self._stack_file = stack_file
		self._particle = particle
		self._E0 = E0
		self._beam_current = beam_current
		self._irradiation_length = irradiation_length
		

		self._parse_kwargs(**kwargs)



	def _parse_kwargs(self, **kwargs):
		# self._dE0 = float(kwargs['dE0']) if 'dE0' in kwargs else 0.01*self._E0
		# self._N = int(kwargs['N']) if 'N' in kwargs else 10000
		# self._dp = float(kwargs['dp']) if 'dp' in kwargs else 1.0
		# self._chunk_size = int(kwargs['chunk_size']) if 'chunk_size' in kwargs else int(1E7)
		# self._accuracy = float(kwargs['accuracy']) if 'accuracy' in kwargs else 0.01
		# self._min_steps = int(kwargs['min_steps']) if 'min_steps' in kwargs else 2

		self._compound_cross_sections_csv = str(kwargs['compound_cross_sections_csv']) if 'compound_cross_sections_csv' in kwargs else 'compound_cross_sections.csv'
		self._n_largest_activities = int(kwargs['n_largest_activities']) if 'n_largest_activities' in kwargs else 20
		self._activity_units = str(kwargs['activity_units']) if 'activity_units' in kwargs else str('uCi')
		self._mass_units = str(kwargs['mass_units']) if 'mass_units' in kwargs else str('ug')
		self._enriched_targets = dict(kwargs['enriched_targets']) if 'enriched_targets' in kwargs else {}
		self._cooling_length = float(kwargs['cooling_length']) if 'cooling_length' in kwargs else 24.0
		self._lower_halflife_threshold = float(kwargs['lower_halflife_threshold']) if 'lower_halflife_threshold' in kwargs else 60
		self._show_plots = bool(kwargs['show_plots']) if 'show_plots' in kwargs else False
		self._save_csv = bool(kwargs['save_csv']) if 'save_csv' in kwargs else True
		self._summary_masses = bool(kwargs['summary_masses']) if 'summary_masses' in kwargs else True

		# beam_current = 1e3               # nA
		# irradiation_length = 1*3600        # s  - half-lives are being pulled in in units of s
		# cooling_length = 24.0               # h  - time after EoB to report activities
		# lower_halflife_threshold = 60    # s    - Remove any products which are unreasonably short
		# show_plots = False               # Show plots of the compound-calculated cross sections
		# save_csv = True                  # Export results to csv - MASSIVELY reduces computational time for subsequent runs
		# summary_masses = True            # In addition to activity calculations, calculate masses of all reaction products
		# particle = 'd'					 # Options are 'p',  'd'   (for now)
		# # E0 = 29.1 						 # MeV  -  incident particle energy
		# # E0 = 15.11 						 # MeV  -  incident particle energy
		# E0 = 25.0 						 # MeV  -  incident particle energy
		# compound_cross_sections_csv = 'compound_cross_sections.csv'    # Output file name for the csv containing XS data for compounds.  Generation of this takes 
		#                                                                # much longer than doing the activity/yield calculations, so exporting and loading this files
		#                                                                # speeds up runs, as long as this file contains data for all compounds in your stack
		# n_largest_activities = 20         # Return the n largest (e.g., the largest 3) activities in each foil for summary 
		# activity_units = 'uCi'           # Specifies activity unit output - options are 'Bq', 'kBq', 'MBq', 'GBq', 'uCi', 'mCi', 'Ci'
		# mass_units = 'ug'                # Specifies mass unit output - options are 'fg', 'pg', 'ng', 'ug', 'mg'

	def calc_yields(self):

		# Check to make sure that the requested incident particle has a TENDL library
		if self._particle.lower() == 'p'.lower():
			# Running with incident protons
			particle = 'p'
		elif self._particle.lower() == 'd'.lower():
			# Running with incident deuterons
			self._particle = 'd'
		elif self._particle.lower() == 'n'.lower():
			# Running with incident neutrons
			# self._particle = 'n'
			print('TENDL has (n,x) data, but this calculator only supports transport of incident charged particles!')
			quit()
		else:
			print('Unsupported particle type \"', self._particle, '\" selected.')
			print('Valid incident particles for TENDL data are currently limited to \"p\", \"d\".')
			quit()

		lb = ci.Library('tendl_'+self._particle)       # selection of reaction data library
		st = ci.Stack(self._stack_file, E0=self._E0, particle=self._particle, dE0=(self._E0*0.015), N=1E5, max_steps=100)      # Load pre-defined stack



		# Make sure target enrichments are normalized
		# print(enriched_targets)
		# Remove all isotopes with abundance set to 0.0
		self._enriched_targets = {a:{c:d for c, d in b.items() if d != 0.0} for a, b in self._enriched_targets.items()}
		# print(not bool(enriched_targets))
		# if use_enriched_targets:
		if bool(self._enriched_targets):
			for compound_iterable, sub_dictionary in self._enriched_targets.items():
				# print(sub_dictionary)
				# print(compound_iterable)
				# print(d['Enrichment'])
				# print(sub_dictionary.values())
				factor = 1.0/sum(sub_dictionary.values())
				if factor != 1.0:
					print('Isotope enrichments in compound ', compound_iterable, ' are not normalized.  Renormalizing...')
					# print(factor)
					for enriched_isotope in sub_dictionary:
						# print(enriched_isotope)
						sub_dictionary[enriched_isotope] = sub_dictionary[enriched_isotope] * factor
						# if sub_dictionary[enriched_isotope] == 0.0:
						# 	print('Target isotope ', enriched_isotope, ' was manually assigned abundance ', sub_dictionary[enriched_isotope], ', dropping from enriched targets...')
						# 	print(compound_iterable)
						# 	print(enriched_isotope)
						# 	print(enriched_targets[compound_iterable][enriched_isotope])
						# 	# del(enriched_targets[compound_iterable][enriched_isotope])
						if sub_dictionary[enriched_isotope] < 0.0:
							print('Target isotope ', enriched_isotope, ' was manually assigned abundance ', sub_dictionary[enriched_isotope])
							print('Abundance improperly formed, stopping program!')
							quit()

				else:
					# print('Isotope enrichments in compound ', compound_iterable, ' are normalized.')
					# for enriched_isotope in sub_dictionary:
					# 	print(enriched_isotope)

					continue

		# print(enriched_targets)


		# quit()

		# Debug testing
		# lb = ci.Library('tendl_p')
		# print(ci.Isotope('81MOg'))

		# N_xs_out = ( ci.Reaction('15N(p,x)15Ng').xs * ci.Isotope('15N').abundance*0.01)  
		# O_xs_out = (ci.Reaction('16O(p,x)15Ng').xs * ci.Isotope('16O').abundance*0.01 + ci.Reaction('17O(p,x)15Ng').xs * ci.Isotope('17O').abundance*0.01 )
		# # print(O_xs_out)
		# # print(lb.search(target='18O', product='15Ng'))
		# cm = ci.Compound('Kapton')
		# # print(cm.weights[cm.weights['element']=='N'].atom_weight.values)
		# print((N_xs_out * cm.weights[cm.weights['element']=='N'].atom_weight.values) + (O_xs_out * cm.weights[cm.weights['element']=='O'].atom_weight.values))

		def lin_interp(x, y, i, half):
		    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

		def half_max_x(x, y):
		    half = max(y)/2.0
		    signs = np.sign(np.add(y, -half))
		    zero_crossings = (signs[0:-2] != signs[1:-1])
		    # print('zero_crossings: ', zero_crossings)
		    zero_crossings_i = np.where(zero_crossings)[0]
		    # print('zero_crossings_i: ', zero_crossings_i)
		    lower_bound = lin_interp(x, y, zero_crossings_i[0], half)
		    # print('Lower bound: ', lower_bound)
		    try:
		        upper_bound = lin_interp(x, y, zero_crossings_i[1], half)
		    except IndexError:
		        upper_bound = max(x)
		    
		    # print('Upper Bound: ', upper_bound)
		    return np.array([lower_bound, upper_bound])


		def interp_along_df(df,energy, flux):
			if len(df.index) == 0:
				# print('Empty DataFrame')
				return 0.0
			else:
				# print(df)
				# print(df['XS'].dtypes)
				# print('XS: ', df['XS'].values)
				# old_energy = df.at[0,'Energy']
				old_energy = np.array(df['Energy'].iloc[0])
				# print('Energy (MeV): ', energy)
				# print('Old Energy (MeV): ', old_energy)
				# print('Energy shape: ', np.shape(energy))
				# print('Old Energy shape: ', np.shape(old_energy))
				# print('Flux shape: ' ,np.shape(flux))
				# print('XS shape', np.shape(np.stack(df['XS'].to_numpy())))
				# print('Energy type: ', type(energy))
				# print('Old Energy type: ', type(old_energy))
				# print('Flux type: ' ,type(flux))
				# print('XS type', type(np.stack(df['XS'].to_numpy())))
				f_out = interpolate.interp1d(old_energy, np.stack(df['XS'].to_numpy()), axis=1)
				avg_xs = np.trapz(np.multiply(f_out(energy),flux), axis=1, x=energy)/np.trapz(flux, x=energy)
				avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
				# print('Flux-Averaged XS (mb): ', avg_xs)
				# print('Flux-Averaged Energy (MeV): ', avg_e)
				# print(df['Name'].values)
				E_bins = half_max_x(energy,flux)
				# print('Energy bins ', E_bins)
				# plt.plot(old_energy, *np.stack(df['XS'].to_numpy()) , label=(df['Name'].values))
				if self._show_plots:
					for y_arr, label, y_xs in zip(np.stack(df['XS'].to_numpy()), df['Name'].values, avg_xs):
					    p = plt.semilogy(old_energy, y_arr,  label=label)
					    plt.plot(avg_e, y_xs , 'o', color=p[0].get_color())
					# ax = plt.gca() 
					plt.legend(loc="upper right")
					plt.xlabel('Beam Energy (MeV)')
					plt.ylabel('Cross Section (mb)')
					plt.show()
				return avg_xs




		# Parse unit selection for later output

		if self._activity_units == 'uCi':
			activity_scalar = 3.7E4
		elif self._activity_units == 'Bq':
			activity_scalar = 1
		elif self._activity_units == 'kBq':
			activity_scalar = 1E3
		elif self._activity_units == 'MBq':
			activity_scalar = 1E6
		elif self._activity_units == 'mCi':
			activity_scalar = 3.7E7
		elif self._activity_units == 'GBq':
			activity_scalar = 1E9
		elif self._activity_units == 'Ci':
			activity_scalar = 3.7E10
		else:
			print('Specified activity units not recognized, defaulting to Bq.')
			activity_scalar = 1


		if self._mass_units == 'fg':
			mass_scalar = 1E15
		elif self._mass_units == 'pg':
			mass_scalar = 1E12
		elif self._mass_units == 'ng':
			mass_scalar = 1E9
		elif self._mass_units == 'ug':
			mass_scalar = 1E6
		elif self._mass_units == 'mg':
			mass_scalar = 1E3
		else:
			print('Specified activity units not recognized, defaulting to fg.')
			mass_scalar = 1E15






		# Load stack data for analysis
		# st.plot('Tl')
		data = st.stack
		# print(data)

		# No need to run if we've already exported to csv!
		try:
			# Column structure:  pd.DataFrame(columns = ['Name', 'Compound', 'Product', 'Energy', 'XS',  'Half-Life', 'Subtargets'])
			# 
			# It turns out pandas csv writing adds /n characters throughout a numpy array
			# This strips out the newlines, and parses them in as the intended data type 
			# 
			def converter(instr):
			    return np.fromstring(instr[1:-1],sep=' ')
			compound_xs_df = pd.read_csv(self._compound_cross_sections_csv, converters={'Energy':converter, 'XS':converter}, dtype={'Name':str, 'Compound':str, 'Product':str,  'Half-Life':np.float64, 'Subtargets':str}, skiprows=1)
			print('Reading compound cross sections from file ' + self._compound_cross_sections_csv)
			# print(data  )
			compound_xs_df =  compound_xs_df[compound_xs_df['Half-Life'] > self._lower_halflife_threshold]
			print(compound_xs_df  )

			rad_products =  compound_xs_df[compound_xs_df['Half-Life'] < np.inf]
			stable_products =  compound_xs_df[compound_xs_df['Half-Life'] == np.inf]

			# print(rad_products)
			# print(stable_products)

			molar_mass_dict = {}

			# print(st.compounds)

			for compound in st.compounds:
				# print(compound)
				cm = ci.Compound(compound)
				# print( cm.weights)
				molar_mass = 0.0
				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					# if use_enriched_targets:
					if bool(self._enriched_targets):	
						# for enriched_isotope_iterator in enriched_targets:
						for compound_iterable, sub_dictionary in self._enriched_targets.items():
							# print('Element index: ',element_index)
							# print('Compound: ',compound)
							if compound.lower() == compound_iterable.lower():
								# print('Subdictionary: ', sub_dictionary)
								# print('Compound iterable: ',compound_iterable)
								# print('Enriched Isotope Iterator: ', enriched_isotope_iterator)
								# print('Element: ',ci.Element(element_row['element']))
								for enriched_isotope_iterator in sub_dictionary:
									# print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
									if enriched_isotope_iterator in ci.Element(element_row['element']).isotopes:
										# print('Compound iterable: ',compound_iterable)
										# # print('Subdictionary: ', sub_dictionary)
										# print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
										# print('Isotope ', enriched_isotope_iterator, ' found in element ', ci.Element(element_row['element']))
										# print('Elemental mass: ', ci.Element(element_row['element']).mass)
										# print('Mass of Isotope ',enriched_isotope_iterator, ' : ', ci.Isotope(enriched_isotope_iterator).mass)
										molar_mass += ci.Isotope(enriched_isotope_iterator).mass * sub_dictionary[enriched_isotope_iterator] * element_row['atom_weight'] #* 0.5
									else:
										molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
							else:
								molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
						
					else:
						molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
					# print(element_row)
				# print(molar_mass)
				molar_mass_dict[compound] = molar_mass
				# print('Compound ', compound, ' has final molar mass of ', molar_mass_dict[compound])
			# print(molar_mass_dict)


			# Get fluxes for each row in compound_df
			# print(compound_df['name'])
			for index, row in data.iterrows():
			    # print(index, row['name'])
				# print(row['compound'], row['name'])
				# print(row)

				rad_indices    = compound_xs_df[(compound_xs_df['Half-Life'] <  np.inf) & (compound_xs_df['Compound'] == row['compound'])].index.values
				stable_indices = compound_xs_df[(compound_xs_df['Half-Life'] == np.inf) & (compound_xs_df['Compound'] == row['compound'])].index.values
				# print(rad_indices, stable_indices)
				# print(len(rad_indices))
				# 
				# 
				# Check for no reactions found for a foil - indicates compound was overlooked, or that compound_cross_sections_csv was generated from a different stack?
				# print('Number of reactions for compound', row['compound'], 'in foil', row['name'], ' :', len( compound_xs_df[(compound_xs_df['Compound'] == row['compound'])].index.values ))
				if len( compound_xs_df[(compound_xs_df['Compound'] == row['compound'])].index.values ) == 0:
					print('No reactions found for compound', row['compound'], 'in foil', row['name'], '.\nThis compound might not exist in ', self._compound_cross_sections_csv, '!\nDelete or rename this file and try running again.')



			    # Column structure: A0_compound_df = pd.DataFrame(columns = ['Name', 'Target', 'Product', 'Energy', 'XS', 'Subreactions'])

				energy, flux = st.get_flux(row['name'])
				# print('Energy: ', energy)
				average_xs = interp_along_df(rad_products[rad_products['Compound'] == row['compound']],energy, flux)
				stable_xs  = interp_along_df(stable_products[stable_products['Compound'] == row['compound']],energy, flux)
				# print('Average XS (mb): ', average_xs)
				# print('Rad Products: ',rad_products.loc[rad_products['Compound'] == row['compound'],'Product'].values)
				# print('Half-life (s) :', rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values)
				# print(molar_mass_dict[row['compound']])
				# print((row['areal_density'] ))
				# print('Production term: ', (1-np.exp(-np.log(2)*irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values )))
				# print('production term: ', (np.exp(-np.log(2)*irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values )))
				print('Compound ', row['compound'], ' has molar mass ',molar_mass_dict[row['compound']])
				A0 = average_xs *  ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]) * self._beam_current * (1-np.exp(-np.log(2)*self._irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values )) * (np.exp(-np.log(2)*self._cooling_length*3600 / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values ))  / (1.602E-10 * 1E27 * activity_scalar)  
				# print('A0 (' + activity_units +'): ', A0)
				rad_mass = A0 * activity_scalar * molar_mass_dict[row['compound']] * (rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values / np.log(2)) * mass_scalar / 6.022E23
				# print('EoB Mass (' + mass_units +'): ', rad_mass)
				# print('Stable products: ', stable_products.loc[stable_products['Compound'] == row['compound'],'Product'].values)
				N_stable = stable_xs * ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]) * self._beam_current * self._irradiation_length   / (1.602E-10 * 1E27 )  
				stable_mass = N_stable * molar_mass_dict[row['compound']] * mass_scalar / 6.022E23
				# print('EoB Stable Mass (' + mass_units +'): ', stable_mass)
				# print(np.shape(rad_mass))
				# 
				# Update the dataframe
				rad_products['A0_'+row['name']] = np.zeros(len(rad_products.index)) 
				rad_products.loc[rad_products['Compound'] == row['compound'],'A0_'+row['name']] = A0
				compound_xs_df['Mass_'+row['name']] = np.zeros(len(compound_xs_df.index)) 
				if version.parse(pd.__version__) < version.parse("2.0.0") :
					compound_xs_df.at[rad_indices,'Mass_'+row['name']] = rad_mass
					compound_xs_df.at[stable_indices,'Mass_'+row['name']] = stable_mass
				else:
					# print(rad_indices)
					# print('Mass_'+row['name'])
					compound_xs_df.loc[rad_indices,'Mass_'+row['name']] = rad_mass
					compound_xs_df.loc[stable_indices,'Mass_'+row['name']] = stable_mass
				
			
			# Pull out largest few activities for summary 
			if self._n_largest_activities > 0:
				# First for activities
				if self._cooling_length == 0:
					summary_string = ",<E> (MeV),<E>-\u03B4E (MeV),<E>+\u03B4E (MeV),Note: all activities are in " + self._activity_units  +' - reported at EoB' + ' ,'*(self._n_largest_activities-1) +'\n'
				else:
					summary_string = ",<E> (MeV),<E>-\u03B4E (MeV),<E>+\u03B4E (MeV),Note: all activities are in " + self._activity_units  +' - reported ' + str(self._cooling_length) + ' h after EoB' + ' ,'*(self._n_largest_activities-1) +'\n'
				for column in rad_products.columns[7:]:
					energy, flux = st.get_flux(column.replace('A0_', '', 1)  )
					avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
					E_bins = half_max_x(energy,flux)
					largest = rad_products.nlargest(self._n_largest_activities, column)
					summary_string +=  column.replace('A0_', '', 1)  + ',' + "{:.2f}".format(avg_e) + ',' + "{:.2f}".format(E_bins[0]) + ',' + "{:.2f}".format(E_bins[1]) 
					for i in np.arange(self._n_largest_activities):
						# summary_string +=  ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ' (' + "{:.2f}".format(largest[column].iloc[i]) + ')' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
						summary_string +=  ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ',' + "{:.2f}".format(largest[column].iloc[i]) + ',' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
					# summary_string.join(i for j in zip(column.strip('A0'), largest['Name'].iloc[0], largest[column].iloc[0],largest['Name'].iloc[1],largest[column].iloc[1],largest['Name'].iloc[2],largest[column].iloc[2]) for i in j)
					# summary_string += str([x for x in itertools.chain(*itertools.zip_longest(column.strip('A0'), largest['Name'].iloc[0], largest[column].iloc[0],largest['Name'].iloc[1],largest[column].iloc[1],largest['Name'].iloc[2],largest[column].iloc[2])) if x is not None])
					summary_string += '\n'
					# print(summary_string)
				text_file = open("activity_summary.csv", "w")
				n = text_file.write(summary_string)
				text_file.close()
				if self._summary_masses:
					# 
					# Repeat for masses
					if self._cooling_length == 0:
						summary_string = ",<E> (MeV),Note: all masses are in " + self._mass_units +' - reported at EoB' + ' ,'*(self._n_largest_activities-1) +'\n'
					else:
						summary_string = ",<E> (MeV),Note: all masses are in " + self._mass_units +' - reported ' + str(self._cooling_length) + ' h after EoB' + ' ,'*(self._n_largest_activities-1) +'\n'
					for column in compound_xs_df.columns[7:]:
						energy, flux = st.get_flux(column.replace('Mass_', '', 1)  )
						avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
						largest = compound_xs_df.nlargest(self._n_largest_activities, column)
						summary_string +=  column.replace('Mass_', '', 1)  + ',' + "{:.2f}".format(avg_e)
						for i in np.arange(self._n_largest_activities):
							# summary_string += ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ' (' + "{:.2e}".format(largest[column].iloc[i]) + ')' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
							summary_string += ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ',' + "{:.2e}".format(largest[column].iloc[i]) + ',' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
							# print(summary_string)
						summary_string += '\n'
					text_file = open("mass_summary.csv", "w")
					n = text_file.write(summary_string)
					text_file.close()






			# Output results to csv
			if self._save_csv:
				rad_products.to_csv('rad_products.csv', index=False)
				compound_xs_df.to_csv('all_product_masses.csv', index=False)

				


		except FileNotFoundError:
			print('File ' + self._compound_cross_sections_csv + ' does not exist - we\'re doing this the hard way')


			# Make empty dataframe to hold reaction data
			reaction_df = pd.DataFrame(columns = ['Name', 'Target', 'Product', 'Energy', 'XS', 'Subreactions'])

			included_elements = []
			from_compound = []


			# Assemble dataframe for elemental (p,x) reactions
			# 
			# for compound in ["Cu"]:
			for compound in st.compounds:

				# print(st.compounds[compound])
				compound_df = data.loc[data['compound'] == compound]
				# print(compound_df)
				cm = ci.Compound(compound)
				# print( cm.weights)

				# # Make empty dataframe to hold reaction data
				# reaction_df = pd.DataFrame(columns = ['Name', 'Target', 'Product', 'Energy', 'XS', 'Subreactions'])


				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					element = element_row['element']
					print(element)
					# Make sure we haven't already added this element to reaction_df
					if element not in included_elements:
						included_elements.append(element)
						from_compound.append(st.compounds[compound])
						em  = ci.Element(element)
						# print('Abundances in element ', element, ':')
						# print(em.abundances)

						if(em.abundances.empty):
							# Target likely has no stable elements
							for compound_iterable, sub_dictionary in self._enriched_targets.items():
									# print('Element index: ',element_index)
									# print('Compound: ',compound)
									if compound.lower() == compound_iterable.lower():
										# Convert enrichment dictionary into a np_array to convert into the pandas df format used by Curie
										# https://stackoverflow.com/questions/30740490/numpy-dtype-for-list-with-mixed-data-types
										enriched_nparray = np.array(list(sub_dictionary.items()), dtype=object)
										# Convert abundances from decimals to percentages
										enriched_nparray[:,1] *= 100
										# Assume zero uncertainty in abundance
										# np.append(enriched_nparray,np.zeros([len(enriched_nparray),1]),1)
										enriched_nparray = np.c_[enriched_nparray, np.zeros(len(enriched_nparray))]
										# print('enriched_nparray: ', enriched_nparray)
							em.abundances = pd.DataFrame(enriched_nparray, columns=['isotope', 'abundance', 'unc_abundance'])
							# print(em.abundances)


						is_compound_enriched = False
						# find all isotopes in the element
						for isotope_index, isotope_row in em.abundances.iterrows():
							# Initialize all abundances to 0.0, just to be safe and avoid issues with enriched isotopes
							abundance = 0.0
							print('Grabbing abundances for isotope ',isotope_row['isotope'],' in element ',element)
							# Use either natural abundances or enriched target abundances
							if bool(self._enriched_targets):
								# if compound in enriched_targets:
								# print('This compound makes use of enriched targets, with abundances: ')
								# print(enriched_targets)
								# are we looking at enriched targets or not
								isotope = isotope_row['isotope']
								for compound_iterable, sub_dictionary in self._enriched_targets.items():
									# print('Element index: ',element_index)
									# print('Compound: ',compound)
									if compound.lower() == compound_iterable.lower():
										# is the current compound part of the enriched isotopes dictionary
										# if compound in enriched_targets:
										print('This compound makes use of enriched targets, with abundances: ')
										print(self._enriched_targets)
										# print('Enriched Isotope Abundances: ', sub_dictionary)
										# print('Compound iterable: ',compound_iterable)
										# print('Element: ',ci.Element(element_row['element']))
										for enriched_isotope_iterator in sub_dictionary:
											print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
											if enriched_isotope_iterator in ci.Element(element_row['element']).isotopes:
												# is the current isotope part of stable abundances
												if enriched_isotope_iterator.lower() == isotope_row['isotope'].lower():
													# then adopt the manually-input abundance rather than natural abundance 
													# 
													# # print('Compound iterable: ',compound_iterable)
													# # print('Subdictionary: ', sub_dictionary)
													# print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
													# # print('Isotope ', enriched_isotope_iterator, ' found in element ', ci.Element(element_row['element']))
													# # print('Elemental mass: ', ci.Element(element_row['element']).mass)
													# print('Mass of Isotope ',enriched_isotope_iterator, ' : ', ci.Isotope(enriched_isotope_iterator).mass)
													abundance = 100 * sub_dictionary[enriched_isotope_iterator]
													is_compound_enriched = True
													continue
											elif enriched_isotope_iterator in sub_dictionary:
												# is the current isotope part of the enriched isotopes dictionary
												if enriched_isotope_iterator.lower() == isotope_row['isotope'].lower():
													# then adopt the manually-input abundance rather than natural abundance 
													# 
													# # print('Compound iterable: ',compound_iterable)
													# # print('Subdictionary: ', sub_dictionary)
													# print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
													# # print('Isotope ', enriched_isotope_iterator, ' found in element ', ci.Element(element_row['element']))
													# # print('Elemental mass: ', ci.Element(element_row['element']).mass)
													# print('Mass of Isotope ',enriched_isotope_iterator, ' : ', ci.Isotope(enriched_isotope_iterator).mass)
													abundance = 100 * sub_dictionary[enriched_isotope_iterator]
													is_compound_enriched = True
													continue
									else:
										# Curie abundances are listed as %, not decimal
										print('This compound does not make use of enriched targets, grabbing natural abundances instead')
										abundance = isotope_row['abundance']
										is_compound_enriched = False
							else:
								# Curie abundances are listed as %, not decimal
								abundance = isotope_row['abundance']
								isotope = isotope_row['isotope']
								is_compound_enriched = False
							print(isotope, abundance)


							if abundance <= 0.0:
								print('Target isotope ', isotope, ' has apparent abundance of ', abundance, ', please double check!!!')

							# pull all available reactions on the target isotope
							list_of_rxns = lb.search(target=isotope)
							# print(list_of_rxns)
							# print('# of Reactions: ', len(list_of_rxns))
							i=0
							# for rxn in list_of_rxns[-10:]:       # Debug mode testing on small number of channels  
							for rxn in list_of_rxns:
								# product = rxn.replace(isotope, "").replace("(p,x)", "")
								product = rxn.strip(isotope).strip("("+self._particle+",x)")
								# print(product)
								# 
								# TENDL apparently has XS data for products which go beyond the proton drip line
								# This catches null values from Curie failing to find them
								# 
								try:
									print(product)
									if ci.Isotope(product).half_life('s') >= self._lower_halflife_threshold:
										if abundance != 0.0:

											# reaction_name = element+'('+particle+',x)'+product
											target = element

											if is_compound_enriched:
												reaction_name = 'enriched'+element+'('+self._particle+',x)'+product
												# target = 'enriched'+element
											else:
												reaction_name = 'nat'+element+'('+self._particle+',x)'+product
												# target = 'nat'+element
											# if compound == 'Ti':
												# print(rxn, product, reaction_name)

											# check if exists in dataframe or not
											# if reaction_name not in reaction_df['Name'] :
											# print(reaction_name)
											# print(reaction_df.values)
											# print(type(reaction_name))
											# print(type(reaction_df.values))
											# test_bool = [np.any(x in element for x in reaction_name) for element in reaction_df.values]
											# print(test_bool)
											# 
											# Deprecating this due to ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
											# if reaction_name not in reaction_df.values :
											if any(element in reaction_name for element in reaction_df.values[:,0]):
												# print("\nThis value exists in Dataframe", reaction_name)
												i+=1
												update_index = reaction_df[reaction_df['Name']==reaction_name].index.tolist()
												# print(update_index[0])
												reaction_df.at[update_index[0],'XS']= reaction_df.at[update_index[0],'XS'] + ci.Reaction(rxn).xs*abundance*0.01
												reaction_df.at[update_index[0],'Subreactions']= reaction_df.at[update_index[0],'Subreactions'] + ',' + isotope
											else:
												# print("\nThis value does not exist in Dataframe", reaction_name)
												if version.parse(pd.__version__) < version.parse("2.0.0") :
													reaction_df = reaction_df.append({'Name' : reaction_name, 'Target' : target, 'Product' : product, 'Energy' : ci.Reaction(rxn).eng, 'XS' : ci.Reaction(rxn).xs*abundance*0.01, 'Subreactions' : isotope}, ignore_index = True)
												else :
													reaction_df = pd.concat([reaction_df, pd.DataFrame([{'Name' : reaction_name, 'Target' : target, 'Product' : product, 'Energy' : ci.Reaction(rxn).eng, 'XS' : ci.Reaction(rxn).xs*abundance*0.01, 'Subreactions' : isotope}])], ignore_index=True)

								except KeyError:
									# pass
									# return np.nan
									print('Curie appears to be missing data for isotope ', product)
								except IndexError:
									# pass
									# return np.nan
									print('Curie appears to be missing data for isotope ', product)
								except TypeError:
									print('Curie appears to be missing data for isotope ', product, ' float() argument must be a string or a number, not ''NoneType''')
			
					# Skip duplicates 
					else :
						# print(included_elements)
						# print(from_compound)
						index = included_elements.index(element)
						print('Element ', element, ' in ', st.compounds[compound], ' already added in compound ', from_compound[index], ', skipping...')


				

			print(reaction_df)
			# Export elemental reaction XS data to csv
			if self._save_csv:
				with open('reaction_df.csv', 'w') as f:
					if bool(self._enriched_targets):
						f.write('Note: all XS calculated using target enrichment of '+str(self._enriched_targets).replace(",","")+',,,,,\n')
						reaction_df.to_csv(f, index=False, mode='a')
					else:
						f.write('Note: all XS calculated using natural abundance targets ,,,,,\n')
						reaction_df.to_csv(f, index=False, mode='a')
					# reaction_df.to_csv(f, index=False, mode='a')
				# reaction_df_file = open('test.csv', 'a')
				# reaction_df_file.write('# My awesome comment,,,,,\n')
				# # reaction_df_file.close()
				# # reaction_df.to_csv('reaction_df.csv', index=False)
				# reaction_df.to_csv('reaction_df.csv', index=False, mode='a')
				# # reaction_df_file.close()
			# print(reaction_df['Subreactions'])


			print('**************************************************************************************************************')
			print('Starting calculation of compound reactions')
			print('**************************************************************************************************************')

			# Make empty dataframe to hold compound reaction data
			compound_rxn_df = pd.DataFrame(columns = ['Name', 'Compound', 'Product', 'Energy', 'XS',  'Half-Life', 'Subtargets'])
			# print(compound_rxn_df)


			for compound in st.compounds:

				print(st.compounds[compound])
				# compound_df = data[data['compound'].str.match(compound)]
				compound_df = data.loc[data['compound'] == compound]
				# print(compound_df)
				cm = ci.Compound(compound)
				rows, cols = np.shape(cm.weights)
				# print( rows, cols)
				# print('Compound name: ', cm.name)




				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					element = element_row['element']
					atom_weight = element_row['atom_weight']
					# print(element)
					# test_df = reaction_df[reaction_df['Target'].str.match('nat'+element)]
					test_df = reaction_df.loc[(reaction_df['Target'] == element)]
					# test_df = reaction_df.loc[(reaction_df['Target'] == 'nat'+element) | (reaction_df['Target'] == 'enriched'+element)]
					

					# for rxn_string in reaction_name:
					for rxn_index, rxn_row in test_df.iterrows():
						rxn_string = cm.name+'('+self._particle+',x)'+rxn_row['Product']
						# print(rxn_string)
						# 
						# check if exists in dataframe or not
						# if reaction_name not in reaction_df['Name'] :
						# 
						# Deprecating this due to ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
						if not any(element in rxn_string for element in compound_rxn_df.values[:,0]):
						# if rxn_string not in compound_rxn_df.values :
							# print("\nThis value does not exist in Dataframe", rxn_string)
							# print(rxn_row['Product'])
							# 
							# TENDL apparently has XS data for products which go beyond the proton drip line
							# This catches null values from Curie failing to find them
							# 
							try: 
								# print(ci.Isotope(rxn_row['Product']))
								# print(ci.Isotope(rxn_row['Product']).half_life('s') )
								if version.parse(pd.__version__) < version.parse("2.0.0") :
									compound_rxn_df = compound_rxn_df.append({'Name' : rxn_string, 'Compound' : cm.name, 'Product' : rxn_row['Product'], 'Energy' : rxn_row['Energy'], 'XS' : rxn_row['XS']*atom_weight, 'Half-Life' : ci.Isotope(rxn_row['Product']).half_life('s') , 'Subtargets' : element}, ignore_index = True)
								else :
									compound_rxn_df = pd.concat([compound_rxn_df, pd.DataFrame([{'Name' : rxn_string, 'Compound' : cm.name, 'Product' : rxn_row['Product'], 'Energy' : rxn_row['Energy'], 'XS' : rxn_row['XS']*atom_weight, 'Half-Life' : ci.Isotope(rxn_row['Product']).half_life('s') , 'Subtargets' : element}])], ignore_index=True)
							except IndexError:
								print('Curie appears to be missing data for isotope ', rxn_row['Product'])
								# compound_rxn_df = compound_rxn_df.append({'Name' : rxn_string, 'Compound' : cm.name, 'Product' : rxn_row['Product'], 'Energy' : rxn_row['Energy'], 'XS' : rxn_row['XS']*atom_weight, 'Half-Life' : 0.0 , 'Subtargets' : element}, ignore_index = True)
							except TypeError:
								print('Curie appears to be missing data for isotope ', rxn_row['Product'], ' float() argument must be a string or a number, not ''NoneType''')
					    					    	
						 
						else :
						    # print("\nThis value exists in Dataframe", rxn_string)
						    update_index = compound_rxn_df[compound_rxn_df['Name']==rxn_string].index.tolist()
						    compound_rxn_df.at[update_index[0],'XS']= compound_rxn_df.at[update_index[0],'XS'] + rxn_row['XS']*atom_weight
						    compound_rxn_df.at[update_index[0],'Subtargets']= compound_rxn_df.at[update_index[0],'Subtargets'] + ',' + element


			print(compound_rxn_df)

			# Export compound XS data to csv
			if self._save_csv:
				with open(self._compound_cross_sections_csv, 'w') as f:
					if bool(self._enriched_targets):
						f.write('Note: all XS calculated using target enrichment of '+str(self._enriched_targets).replace(",","")+',,,,,\n')
						compound_rxn_df.to_csv(f, index=False, mode='a')
					else:
						f.write('Note: all XS calculated using natural abundance targets ,,,,,\n')
						compound_rxn_df.to_csv(f, index=False, mode='a')
					# compound_rxn_df.to_csv(f, index=False, mode='a')
					# compound_rxn_df.to_csv(compound_cross_sections_csv, index=False)
				print('XS data for all compounds in this stack have been saved to ', self._compound_cross_sections_csv, '.  Run this script again to generate activity and yield estimates.')



