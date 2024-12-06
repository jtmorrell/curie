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
	Stack class to perform transport calculations, based on TENDL data from the curie,Library() 
	class. Yields are reported for both stable and radionuclide production for both EOB yields 
	as well as after a user-specified decay time post-EOB. However, all yields are calculated 
	as independent yields, i.e., no contributions from decay feeding are accounted for at present.
	
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
		Incident ion.  For light ions, options are currently 'p' (default), 'd',  for proton, 
		deuteron, respectively, based on TENDL data available in Curie.  Additionally, heavy ions can be
		specified either by element or isotope, e.g. 'Fe', '40CA', 'U', 'Bi-209'.For 
		light ions, the charge state is assumed to be fully stripped. For heavy ions
		the charge state is handled by a Bohr/Northcliffe parameterization consistent
		with the Anderson-Ziegler formalism. 

	E0 : float
		Incident particle energy, in MeV. E0 defaults to 30.0 MeV. If dE0 is not provided, it will
		default to 1 percent of E0.

	beam_current : float
		Incident beam intensity, in particle nA (pnA). For beam currents in electrical nA (enA), 
		multiply by the charge state of the beam ions.  For example, for a 2+ charge state He 
		beam (He 2+), 1 pnA = 2 enA. Defaults to 1E3 nA.  

	irradiation_length : float
		The length of the irradiation, in s. Yield calculations assume that the input beam 
		current remains constant for this duration of irradiation.  Defaults to 3600.0 s.  


	Other Parameters
	----------------
	compound_cross_sections_csv : str, optional
		Output file name for the .csv containing calculated data for compounds and non-monoisotopic 
		targets. These are calculated from the TENDL monoisotopic target residual product cross 
		section data during initial execution of Yield.calculate_yields(), and is the dominant source 
		of computational time for yield calculations. Generation of this database takes much longer 
		than doing the actual activity/yield calculations, so exporting and loading this file 
		speeds up future runs, as long as this file contains data for all compounds in an input 
		stack. This parameter also specifies the name of the file that Yield looks to read from in 
		these subsequent runs. Defaults to 'compound_cross_sections.csv'.

		Note: in future versions of Curie.Yield(), natural abundance cross sections will be 
		pre-calculated from TENDL data. This will also add a user option to save calculated
		cross sections for compounds to the user's local database.

	enriched_targets : dict, optional
		By default, yield calculations assume that all elements in stack compounds are present in their 
		natural abundances. This parameter overrides these natural abundances for specified elements 
		only, with a dictionary of enriched targets. Enrichments are specified in atom-basis fractions. 
		These Will be automatically renormalized if the specified enrichments do not sum to 1.0. 
		This supports radionuclides, and this this parameter can be used to run yields on radioactive 
		targets. As these enrichments are used in generating the cross sections for compounds, the 
		compound_cross_sections_csv file must be regenerated when changing target enrichment.   
		Defaults to enriched_targets = {}.

		Dicts must be formatted as {'Element': {'Isotope': Enrichment}}, and an empty dict will default 
		to enriched_targets = {}. Examples of enriched_targets include:

		enriched_targets = {} - this assumes all elements in the stack are natural abundance.

		enriched_targets = {'Yb': {'176YB': 1.0}} - this assumes that all Ytterbium in the stack is 
		100% 176-Yb.

		enriched_targets = {'W': {'180W': 0.0012, '182W': 0.2650, '183W': 0.1431, '184W': 0.3064, 
		'186W': 0.2843}} - this assumes that all Tungsten in the stack is 0.12% 180-W, 26.50% 182-W, 
		14.31% 183-W, and 30.64% 184-W - as this is the natural isotopic composition of W, the results
		will be be identical to enriched_targets = {}.

		enriched_targets = {'Ni': {'58NI': 3, '62NI': 1}} - this assumes all Nickel in the stack is 
		3 parts 58-Ni to 1 part 62-Ni, and will be automatically renormalzied to enriched_targets = 
		{'Ni': {'58NI': 0.75, '62NI': 0.25}}.


	cooling_length : float, optional
		The duration of time after EOB at which yields are reported, in h. cooling_length = 0 will 
		instead report all yields in saved reports at EOB. Defaults to 24.0 h.

	lower_halflife_threshold : float, optional
		A lower threshold for the half-life of product isotopes to be reported. All radionuclides
		with half-lives shorter than lower_halflife_threshold will be excluded from generated reports, 
		useful for removing the many radionuclides and isomers whose lifetimes are unreasonably
		short for analysis of predicted yields.  Defaults to 60.0 s.

	n_largest_products : int, optional
		Truncates the output summary reports to return the n largest (e.g., the largest 3) activities 
		in each foil for activity summaries, and the n largest masses for stable isotope production 
		summaries. Defaults to 5.

	activity_units : str, optional
		Specifies the preferred units for activity in output reports - options are 'Bq', 'kBq', 'MBq', 
		'GBq', 'nCi', 'uCi', 'mCi', 'Ci'. Defaults to 'uCi'.

	mass_units : str, optional
		Specifies the preferred units for mass (of both stable and radionuclide products) in output 
		reports - options are 'fg', 'pg', 'ng', 'ug', 'mg'. Defaults to 'ug'.

	save_csv : bool, optional
		This flag (when save_csv = True) causes the results of calculated cross sections for all stack 
		compounds to be saved to a series of local .csv files after execution. This is highly 
		recommended, particularly for the compound_cross_sections_csv optional parameter, as reading these 
		cross sections from this file MASSIVELY reduces computational time for subsequent runs. The file 
		must be regenerated if new compounds are added to the stack, or if the isotopic enrichment of 
		any materials changes for subsequent runs. Defaults to True.

	show_plots : bool, optional
		When generating the cross sections for stack compounds, this flag (when show_plots = True) 
		enables the display of the excitation function plots of the compound-calculated cross sections. 
		Defaults to False.

	summary_masses : bool, optional
		By default, Yield.calculate_yields() generates summary reports for the EOB yield activity for 
		all radionuclide products with half-lives longer than lower_halflife_threshold, This flag (when 
		summary_masses = True) also generates reports for the mass of both stable and radionuclide 
		products. Defaults to True.


	Attributes
	----------
	reaction_df : pd.DataFrame
		Table of cross sections for reactions on individual elements (either enriched or natural
		abundance), calculated from monoisotopic targets based on the stack design.  Saves locally as 
		'reaction_df.csv'.

	compound_rxn_df : pd.DataFrame
		Table of cross sections for reactions on individual compounds, calculated from the elements in the 
		reaction_df dataframe, based on the stack design.  Saves locally using the filename specified in
		the compound_cross_sections_csv optional parameter.

	product_activities : pd.DataFrame
		Table of all radionuclide activities produced in each stack layer, calculated at time EOB + 
		cooling_length, in user-specified units. Saves locally as 'rad_product_activities.csv'.

	product_masses : pd.DataFrame
		Table of masses of all products (both stable and radioactive) produced in each stack layer, 
		calculated at time EOB + cooling_length, in user-specified units. Saves locally as 
		'all_product_masses.csv'.


	
	

	"""

	def __init__(self, stack_file, particle='p', E0=30.0, beam_current=1e3, irradiation_length=3600.0, **kwargs):
		self.stack_file = stack_file
		self.particle = particle
		self.E0 = E0
		self.beam_current = beam_current
		self.irradiation_length = irradiation_length

		self.calculate_neutron_yields = False
		

		self._parse_kwargs(**kwargs)


		self.reaction_df = None
		self.compound_rxn_df = None

		self.product_activities = None
		self.product_masses = None

		
		



	def _parse_kwargs(self, **kwargs):
		self.compound_cross_sections_csv = str(kwargs['compound_cross_sections_csv']) if 'compound_cross_sections_csv' in kwargs else 'compound_cross_sections.csv'
		self.n_largest_products = int(kwargs['n_largest_products']) if 'n_largest_products' in kwargs else 5
		self.activity_units = str(kwargs['activity_units']) if 'activity_units' in kwargs else str('uCi')
		self.mass_units = str(kwargs['mass_units']) if 'mass_units' in kwargs else str('ug')
		self.enriched_targets = dict(kwargs['enriched_targets']) if 'enriched_targets' in kwargs else {}
		self.cooling_length = float(kwargs['cooling_length']) if 'cooling_length' in kwargs else 24.0
		self.lower_halflife_threshold = float(kwargs['lower_halflife_threshold']) if 'lower_halflife_threshold' in kwargs else 60
		self.show_plots = bool(kwargs['show_plots']) if 'show_plots' in kwargs else False
		self.save_csv = bool(kwargs['save_csv']) if 'save_csv' in kwargs else True
		self.summary_masses = bool(kwargs['summary_masses']) if 'summary_masses' in kwargs else True

		


		# Check to make sure that the requested incident particle has a TENDL library
		if self.particle.lower() == 'p'.lower():
			# Running with incident protons
			particle = 'p'
		elif self.particle.lower() == 'd'.lower():
			# Running with incident deuterons
			self.particle = 'd'
		elif self.particle.lower() == 'n'.lower():
			# Running with incident neutrons
			# self.particle = 'n'
			self.calculate_neutron_yields = True
			print('TENDL has (n,x) data, but this calculator only supports transport of incident charged particles!')
			# print('Neutron compound', self.stack_file["compound"])
			# print('Neutron solid_angle', self.stack_file["solid_angle"])
			# print('Neutron spectrum', self.stack_file["neutron_spectrum"])
			# print('Neutron stack_file', self.stack_file["stack_file"])
			# print(self.stack_file)
			# print(self.calculate_neutron_yields)
			# quit()
		else:
			print('Unsupported particle type \"'+str(self.particle)+'\" selected.')
			print('Valid incident particles for TENDL data are currently limited to \"p\", \"d\".')
			quit()

	def calculate_yields(self):

		

		lb = ci.Library('tendl_'+self.particle)       # selection of reaction data library
		if not self.calculate_neutron_yields:
			st = ci.Stack(self.stack_file, E0=self.E0, particle=self.particle, dE0=(self.E0*0.015), N=1E5, max_steps=100)      # Load pre-defined stack
		else:
			print(self.stack_file["compound"])
			print(type(self.stack_file["compound"]))
			# print(self.stack_file["compound"].weights)
			# print(type(self.stack_file["compound"].weights))
			st = ci.Stack(self.stack_file["stack_file"], compounds=self.stack_file["compound_weights"], E0=self.E0, particle=self.particle, dE0=(self.E0*0.015), N=1E5, max_steps=100)      # Run dummy stack


		# Make sure target enrichments are normalized
		# Remove all isotopes with abundance set to 0.0
		self.enriched_targets = {a:{c:d for c, d in b.items() if d != 0.0} for a, b in self.enriched_targets.items()}
		# if use_enriched_targets:
		if bool(self.enriched_targets):
			for compound_iterable, sub_dictionary in self.enriched_targets.items():
				factor = 1.0/sum(sub_dictionary.values())
				if factor != 1.0:
					print('Isotope enrichments in compound ', compound_iterable, ' are not normalized.  Renormalizing...')
					for enriched_isotope in sub_dictionary:
						sub_dictionary[enriched_isotope] = sub_dictionary[enriched_isotope] * factor
						if sub_dictionary[enriched_isotope] < 0.0:
							print('Target isotope ', enriched_isotope, ' was manually assigned abundance ', sub_dictionary[enriched_isotope])
							print('Abundance improperly formed, stopping program!')
							quit()

				else:
					continue


		def lin_interp(x, y, i, half):
		    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))



		def half_max_x(x, y):
		    half = max(y)/2.0
		    signs = np.sign(np.add(y, -half))
		    zero_crossings = (signs[0:-2] != signs[1:-1])
		    zero_crossings_i = np.where(zero_crossings)[0]
		    lower_bound = lin_interp(x, y, zero_crossings_i[0], half)
		    try:
		        upper_bound = lin_interp(x, y, zero_crossings_i[1], half)
		    except IndexError:
		        upper_bound = max(x)
		    
		    return np.array([lower_bound, upper_bound])


		def interp_along_df(df,energy, flux):
			if len(df.index) == 0:
				return 0.0
			else:
				# print('flux:', flux)
				old_energy = np.array(df['Energy'].iloc[0])
				# f_out = interpolate.interp1d(old_energy, np.stack(df['XS'].to_numpy()), axis=1)
				if self.calculate_neutron_yields:
					num_rows_in_df = df.shape[0]
					avg_xs = np.zeros(num_rows_in_df)
					i=0
					for index, row in df.iterrows():
						# print(row['c1'], row['c2'])
						# print('index:', index )
						# print(type(row['XS']))
						# print('row xs: ', row['XS'])

						f_out = interpolate.interp1d(row['Energy'], row['XS'])
						# print('f_out:', f_out(energy))
						avg_xs[i] = np.trapz(np.multiply(f_out(energy),flux),  x=energy)
						i=i+1
					# print(df)
					# print(df.columns.tolist())
					# print(df["XS"])
					# plt.plot(old_energy,df.iloc[24,4], label='TENDL')
					# plt.plot(energy, f_out(energy)[24,:], label='Interpolated')
					# plt.plot(15.5, avg_xs[24], marker='o', label='avg')
					# plt.xlabel('Energy (MeV)')
					# plt.ylabel('XS')
					# plt.legend()
					# plt.show()
					# print(avg_xs[24])
				else:
					f_out = interpolate.interp1d(old_energy, np.stack(df['XS'].to_numpy()), axis=1)
					avg_xs = np.trapz(np.multiply(f_out(energy),flux), axis=1, x=energy)/np.trapz(flux, x=energy)
				avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
				E_bins = half_max_x(energy,flux)
				if self.show_plots:
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

		if self.activity_units == 'nCi':
			activity_scalar = 3.7E1
		elif self.activity_units == 'uCi':
			activity_scalar = 3.7E4
		elif self.activity_units == 'Bq':
			activity_scalar = 1
		elif self.activity_units == 'kBq':
			activity_scalar = 1E3
		elif self.activity_units == 'MBq':
			activity_scalar = 1E6
		elif self.activity_units == 'mCi':
			activity_scalar = 3.7E7
		elif self.activity_units == 'GBq':
			activity_scalar = 1E9
		elif self.activity_units == 'Ci':
			activity_scalar = 3.7E10
		else:
			print('Specified activity units not recognized, defaulting to Bq.')
			activity_scalar = 1


		if self.mass_units == 'fg':
			mass_scalar = 1E15
		elif self.mass_units == 'pg':
			mass_scalar = 1E12
		elif self.mass_units == 'ng':
			mass_scalar = 1E9
		elif self.mass_units == 'ug':
			mass_scalar = 1E6
		elif self.mass_units == 'mg':
			mass_scalar = 1E3
		else:
			print('Specified activity units not recognized, defaulting to fg.')
			mass_scalar = 1E15






		# Load stack data for analysis
		data = st.stack

		# No need to run if we've already exported to csv!
		try:
			# Column structure:  pd.DataFrame(columns = ['Name', 'Compound', 'Product', 'Energy', 'XS',  'Half-Life', 'Subtargets'])
			# 
			# It turns out pandas csv writing adds /n characters throughout a numpy array
			# This strips out the newlines, and parses them in as the intended data type 
			# 
			def converter(instr):
			    return np.fromstring(instr[1:-1],sep=' ')
			compound_xs_df = pd.read_csv(self.compound_cross_sections_csv, converters={'Energy':converter, 'XS':converter}, dtype={'Name':str, 'Compound':str, 'Product':str,  'Half-Life':np.float64, 'Subtargets':str}, skiprows=1)
			print('Reading compound cross sections from file ' + self.compound_cross_sections_csv)
			self.compound_rxn_df = compound_xs_df
			# Getter to load reaction_df
			self.reaction_df = pd.read_csv('reaction_df.csv', converters={'Energy':converter, 'XS':converter}, dtype={'Name':str, 'Target':str, 'Product':str,  'Subreactions':str}, skiprows=1)
			compound_xs_df =  compound_xs_df[compound_xs_df['Half-Life'] > self.lower_halflife_threshold]
			print(compound_xs_df  )

			rad_products =  compound_xs_df[compound_xs_df['Half-Life'] < np.inf]
			stable_products =  compound_xs_df[compound_xs_df['Half-Life'] == np.inf]

			molar_mass_dict = {}

			for compound in st.compounds:
				if self.calculate_neutron_yields:
					cm = self.stack_file["compound"]
				else:
					cm = ci.Compound(compound)
				molar_mass = 0.0
				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					# if use_enriched_targets:
					if bool(self.enriched_targets):	
						# for enriched_isotope_iterator in enriched_targets:
						for compound_iterable, sub_dictionary in self.enriched_targets.items():
							if compound.lower() == compound_iterable.lower():
								for enriched_isotope_iterator in sub_dictionary:
									if enriched_isotope_iterator in ci.Element(element_row['element']).isotopes:
										molar_mass += ci.Isotope(enriched_isotope_iterator).mass * sub_dictionary[enriched_isotope_iterator] * element_row['atom_weight'] #* 0.5
									else:
										molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
							else:
								molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
						
					else:
						molar_mass += ci.Element(element_row['element']).mass * element_row['atom_weight']
				molar_mass_dict[compound] = molar_mass



			# Get fluxes for each row in compound_df
			for index, row in data.iterrows():

				rad_indices    = compound_xs_df[(compound_xs_df['Half-Life'] <  np.inf) & (compound_xs_df['Compound'] == row['compound'])].index.values
				stable_indices = compound_xs_df[(compound_xs_df['Half-Life'] == np.inf) & (compound_xs_df['Compound'] == row['compound'])].index.values
				# 
				# 
				# Check for no reactions found for a foil - indicates compound was overlooked, or that compound_cross_sections_csv was generated from a different stack?
				if len( compound_xs_df[(compound_xs_df['Compound'] == row['compound'])].index.values ) == 0:
					print('No reactions found for compound', row['compound'], 'in foil', row['name'], '.\nThis compound might not exist in ', self.compound_cross_sections_csv, '!\nDelete or rename this file and try running again.')



				# Column structure: A0_compound_df = pd.DataFrame(columns = ['Name', 'Target', 'Product', 'Energy', 'XS', 'Subreactions'])

				if not self.calculate_neutron_yields:
					energy, flux = st.get_flux(row['name'])
				else:
					# print('energy: ',self.stack_file["neutron_spectrum"][:,0].data)
					# print(type(self.stack_file["neutron_spectrum"][:,0]).data)
					energy = self.stack_file["neutron_spectrum"][:,0].data
					flux = self.stack_file["neutron_spectrum"][:,1].data
					if self.show_plots:
						plt.plot(energy, flux)
						plt.xlabel('Neutron Energy (MeV)')
						plt.ylabel('Neutron Yield (n/MeV/uC/Sr)')
						plt.show()
					# energy, flux = self.stack_file["neutron_spectrum"][0,:], self.stack_file["neutron_spectrum"][1,:]
				average_xs = interp_along_df(rad_products[rad_products['Compound'] == row['compound']],energy, flux)
				stable_xs  = interp_along_df(stable_products[stable_products['Compound'] == row['compound']],energy, flux)
				print('Compound ', row['compound'], ' has molar mass ',molar_mass_dict[row['compound']])
				if not self.calculate_neutron_yields:
					A0 = average_xs *  ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]) * self.beam_current * (1-np.exp(-np.log(2)*self.irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values )) * (np.exp(-np.log(2)*self.cooling_length*3600 / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values ))  / (1.602E-10 * 1E27 * activity_scalar)  
				else:
					# avg_xs * d_current * solid_angle * areal density * unit_conv * saturation_term 
					# print('average xs: ',average_xs[24])
					# print('solid angle',self.stack_file["solid_angle"])
					# print('rho r', ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]))
					# print('beam current', self.beam_current)
					# print('production term', (1-np.exp(-np.log(2)*self.irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values ))[24])
					A0 = average_xs * self.stack_file["solid_angle"] * ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]) * self.beam_current * 1E-3 * (1-np.exp(-np.log(2)*self.irradiation_length / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values )) * (np.exp(-np.log(2)*self.cooling_length*3600 / rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values ))  / (1E27 * activity_scalar)  
					# print('A0', A0[24])
				rad_mass = A0 * activity_scalar * molar_mass_dict[row['compound']] * (rad_products.loc[rad_products['Compound'] == row['compound'],'Half-Life'].values / np.log(2)) * mass_scalar / 6.022E23
				N_stable = stable_xs * ( (row['areal_density'] ) * 6.022E20 / molar_mass_dict[row['compound']]) * self.beam_current * self.irradiation_length   / (1.602E-10 * 1E27 )  
				stable_mass = N_stable * molar_mass_dict[row['compound']] * mass_scalar / 6.022E23
				# 
				# Update the dataframe
				rad_products['A0_'+row['name']] = np.zeros(len(rad_products.index)) 
				rad_products.loc[rad_products['Compound'] == row['compound'],'A0_'+row['name']] = A0
				compound_xs_df['Mass_'+row['name']] = np.zeros(len(compound_xs_df.index)) 
				if version.parse(pd.__version__) < version.parse("2.0.0") :
					compound_xs_df.at[rad_indices,'Mass_'+row['name']] = rad_mass
					compound_xs_df.at[stable_indices,'Mass_'+row['name']] = stable_mass
				else:
					compound_xs_df.loc[rad_indices,'Mass_'+row['name']] = rad_mass
					compound_xs_df.loc[stable_indices,'Mass_'+row['name']] = stable_mass
				
			
			# Pull out largest few activities for summary 
			if self.n_largest_products > 0:
				# First for activities
				if self.cooling_length == 0:
					summary_string = ",<E> (MeV),<E>-\u03B4E (MeV),<E>+\u03B4E (MeV),Note: all activities are in " + self.activity_units  +' - reported at EoB' + ' ,'*(self.n_largest_products-1) +'\n'
				else:
					summary_string = ",<E> (MeV),<E>-\u03B4E (MeV),<E>+\u03B4E (MeV),Note: all activities are in " + self.activity_units  +' - reported ' + str(self.cooling_length) + ' h after EoB' + ' ,'*(self.n_largest_products-1) +'\n'
				for column in rad_products.columns[7:]:
					if not self.calculate_neutron_yields:
						energy, flux = st.get_flux(column.replace('A0_', '', 1)  )
					else:
						energy = self.stack_file["neutron_spectrum"][:,0].data
						flux = self.stack_file["neutron_spectrum"][:,1].data
					avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
					E_bins = half_max_x(energy,flux)
					largest = rad_products.nlargest(self.n_largest_products, column)
					print(largest)
					print(self.n_largest_products)
					summary_string +=  column.replace('A0_', '', 1)  + ',' + "{:.2f}".format(avg_e) + ',' + "{:.2f}".format(E_bins[0]) + ',' + "{:.2f}".format(E_bins[1]) 
					for i in np.arange(self.n_largest_products):
						summary_string +=  ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ',' + "{:.2f}".format(largest[column].iloc[i]) + ',' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
					summary_string += '\n'
				text_file = open("activity_summary.csv", "w")
				n = text_file.write(summary_string)
				text_file.close()
				if self.summary_masses:
					# 
					# Repeat for masses
					if self.cooling_length == 0:
						summary_string = ",<E> (MeV),Note: all masses are in " + self.mass_units +' - reported at EoB' + ' ,'*(self.n_largest_products-1) +'\n'
					else:
						summary_string = ",<E> (MeV),Note: all masses are in " + self.mass_units +' - reported ' + str(self.cooling_length) + ' h after EoB' + ' ,'*(self.n_largest_products-1) +'\n'
					for column in compound_xs_df.columns[7:]:
						if not self.calculate_neutron_yields:
							energy, flux = st.get_flux(column.replace('Mass_', '', 1)  )
						else:
							energy = self.stack_file["neutron_spectrum"][:,0].data
							flux = self.stack_file["neutron_spectrum"][:,1].data
						avg_e = np.trapz(np.multiply(energy,flux), x=energy)/np.trapz(flux, x=energy)
						largest = compound_xs_df.nlargest(self.n_largest_products, column)
						summary_string +=  column.replace('Mass_', '', 1)  + ',' + "{:.2f}".format(avg_e)
						for i in np.arange(self.n_largest_products):
							summary_string += ',' + largest['Product'].iloc[i].replace('g', '', 1) +  ',' + "{:.2e}".format(largest[column].iloc[i]) + ',' #,' + largest['Product'].iloc[1].replace('g', '', 1) + ', (' +  str(largest[column].iloc[1]) + '),' +  largest['Product'].iloc[2] + ', (' +  str(largest[column].iloc[2]) + '\n'
						summary_string += '\n'
					text_file = open("mass_summary.csv", "w")
					n = text_file.write(summary_string)
					text_file.close()



			# Update class attributes with results
			self.product_activities = rad_products
			self.product_masses = compound_xs_df


			# Output results to csv
			if self.save_csv:
				self.product_activities.to_csv('rad_product_activities.csv', index=False)
				self.product_masses.to_csv('all_product_masses.csv', index=False)

				


		except FileNotFoundError:
			print('File ' + self.compound_cross_sections_csv + ' does not exist - we\'re doing this the hard way')


			# Make empty dataframe to hold reaction data
			reaction_df = pd.DataFrame(columns = ['Name', 'Target', 'Product', 'Energy', 'XS', 'Subreactions'])

			included_elements = []
			from_compound = []


			# Assemble dataframe for elemental (p,x) reactions
			# 
			for compound in st.compounds:

				compound_df = data.loc[data['compound'] == compound]
				if self.calculate_neutron_yields:
					cm = self.stack_file["compound"]
				else:
					cm = ci.Compound(compound)

				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					element = element_row['element']
					print(element)
					# Make sure we haven't already added this element to reaction_df
					if element not in included_elements:
						included_elements.append(element)
						from_compound.append(st.compounds[compound])
						em  = ci.Element(element)

						if(em.abundances.empty):
							# Target likely has no stable elements
							for compound_iterable, sub_dictionary in self.enriched_targets.items():
									if compound.lower() == compound_iterable.lower():
										# Convert enrichment dictionary into a np_array to convert into the pandas df format used by Curie
										# https://stackoverflow.com/questions/30740490/numpy-dtype-for-list-with-mixed-data-types
										enriched_nparray = np.array(list(sub_dictionary.items()), dtype=object)
										# Convert abundances from decimals to percentages
										enriched_nparray[:,1] *= 100
										# Assume zero uncertainty in abundance
										enriched_nparray = np.c_[enriched_nparray, np.zeros(len(enriched_nparray))]
							em.abundances = pd.DataFrame(enriched_nparray, columns=['isotope', 'abundance', 'unc_abundance'])


						is_compound_enriched = False
						# find all isotopes in the element
						for isotope_index, isotope_row in em.abundances.iterrows():
							# Initialize all abundances to 0.0, just to be safe and avoid issues with enriched isotopes
							abundance = 0.0
							print('Grabbing abundances for isotope ',isotope_row['isotope'],' in element ',element)
							# Use either natural abundances or enriched target abundances
							if bool(self.enriched_targets):
								# are we looking at enriched targets or not
								isotope = isotope_row['isotope']
								for compound_iterable, sub_dictionary in self.enriched_targets.items():
									if compound.lower() == compound_iterable.lower():
										# is the current compound part of the enriched isotopes dictionary
										print('This compound makes use of enriched targets, with abundances: ')
										print(self.enriched_targets)
										for enriched_isotope_iterator in sub_dictionary:
											print('Enriched Isotope Iterator: ', enriched_isotope_iterator,' has enrichment ',sub_dictionary[enriched_isotope_iterator])
											if enriched_isotope_iterator in ci.Element(element_row['element']).isotopes:
												# is the current isotope part of stable abundances
												if enriched_isotope_iterator.lower() == isotope_row['isotope'].lower():
													# then adopt the manually-input abundance rather than natural abundance 
													abundance = 100 * sub_dictionary[enriched_isotope_iterator]
													is_compound_enriched = True
													continue
											elif enriched_isotope_iterator in sub_dictionary:
												# is the current isotope part of the enriched isotopes dictionary
												if enriched_isotope_iterator.lower() == isotope_row['isotope'].lower():
													# then adopt the manually-input abundance rather than natural abundance 
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
							i=0
							# for rxn in list_of_rxns[-10:]:       # Debug mode testing on small number of channels  
							for rxn in list_of_rxns:
								product = rxn.strip(isotope).strip("("+self.particle+",x)")
								# print(product)
								# 
								# TENDL apparently has XS data for products which go beyond the proton drip line
								# This catches null values from Curie failing to find them
								# 
								try:
									if ci.Isotope(product).half_life('s') >= self.lower_halflife_threshold:
										if abundance != 0.0:
											target = element

											if is_compound_enriched:
												reaction_name = 'enriched'+element+'('+self.particle+',x)'+product
											else:
												reaction_name = 'nat'+element+'('+self.particle+',x)'+product

											# check if exists in dataframe or not
											# 
											# Deprecating this due to ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
											# if reaction_name not in reaction_df.values :
											# print(reaction_df)
											if any(element in reaction_name for element in reaction_df.values[:,0]):
												i+=1
												update_index = reaction_df[reaction_df['Name']==reaction_name].index.tolist()
												# print(ci.Reaction(rxn).xs)
												# print('---------------------------------------------------------------------------')
												# print(np.shape(ci.Reaction(rxn).xs)[0])
												# print(ci.Reaction(rxn))
												# print(abundance)
												# print(self.calculate_neutron_yields)
												if self.calculate_neutron_yields:
													# print(ci.Reaction(rxn))
													# print(ci.Reaction(rxn).xs)
													if np.shape(ci.Reaction(rxn).xs)[0] != np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0]:
														# print('Product ',ci.Reaction(rxn), ' has ', np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0], 'energy points, adding interpolated XS data with ',np.shape(ci.Reaction(rxn).xs)[0], ' more...')
														f_out = interpolate.interp1d(ci.Reaction(rxn).eng, ci.Reaction(rxn).xs)
														reaction_df.at[update_index[0],'XS']= reaction_df.at[update_index[0],'XS'] + f_out(reaction_df.iloc[update_index[0]]['Energy'])*abundance*0.01
													else:
														# print('Product ',ci.Reaction(rxn), ' has ', np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0], 'energy points, adding interpolated XS data with ',np.shape(ci.Reaction(rxn).xs)[0], ' more...')
														reaction_df.at[update_index[0],'XS']= reaction_df.at[update_index[0],'XS'] + ci.Reaction(rxn).xs*abundance*0.01
												else:
													# print('Product ',ci.Reaction(rxn), ' has ', np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0], 'energy points, adding interpolated XS data with ',np.shape(ci.Reaction(rxn).xs)[0], ' more...')
													reaction_df.at[update_index[0],'XS']= reaction_df.at[update_index[0],'XS'] + ci.Reaction(rxn).xs*abundance*0.01

												reaction_df.at[update_index[0],'Subreactions']= reaction_df.at[update_index[0],'Subreactions'] + ',' + isotope
												# print(reaction_df.at[update_index[0],'Subreactions'] + ',' + isotope)
											else:
												if version.parse(pd.__version__) < version.parse("2.0.0") :
													reaction_df = reaction_df.append({'Name' : reaction_name, 'Target' : target, 'Product' : product, 'Energy' : ci.Reaction(rxn).eng, 'XS' : ci.Reaction(rxn).xs*abundance*0.01, 'Subreactions' : isotope}, ignore_index = True)
												else :
													reaction_df = pd.concat([reaction_df, pd.DataFrame([{'Name' : reaction_name, 'Target' : target, 'Product' : product, 'Energy' : ci.Reaction(rxn).eng, 'XS' : ci.Reaction(rxn).xs*abundance*0.01, 'Subreactions' : isotope}])], ignore_index=True)

								except KeyError:
									# pass
									# return np.nan
									print('KeyError: Curie appears to be missing data for isotope ', product)
								except IndexError:
									# pass
									# return np.nan
									print('IndexError: Curie appears to be missing data for isotope ', product)
								except TypeError:
									print('Curie appears to be missing data for isotope ', product, ' float() argument must be a string or a number, not ''NoneType''')
			
					# Skip duplicates 
					else :
						index = included_elements.index(element)
						print('Element ', element, ' in ', st.compounds[compound], ' already added in compound ', from_compound[index], ', skipping...')


			self.reaction_df = reaction_df

			print(reaction_df)
			# Export elemental reaction XS data to csv
			if self.save_csv:
				with open('reaction_df.csv', 'w') as f:
					if bool(self.enriched_targets):
						f.write('Note: all XS calculated using target enrichment of '+str(self.enriched_targets).replace(",","")+',,,,,\n')
						self.reaction_df.to_csv(f, index=False, mode='a')
					else:
						f.write('Note: all XS calculated using natural abundance targets ,,,,,\n')
						self.reaction_df.to_csv(f, index=False, mode='a')



			print('**************************************************************************************************************')
			print('Starting calculation of compound reactions')
			print('**************************************************************************************************************')

			# Make empty dataframe to hold compound reaction data
			compound_rxn_df = pd.DataFrame(columns = ['Name', 'Compound', 'Product', 'Energy', 'XS',  'Half-Life', 'Subtargets'])


			for compound in st.compounds:

				print(st.compounds[compound])
				compound_df = data.loc[data['compound'] == compound]
				if self.calculate_neutron_yields:
					cm = self.stack_file["compound"]
				else:
					cm = ci.Compound(compound)
				rows, cols = np.shape(cm.weights)




				# find all elements in the compound
				for element_index, element_row in cm.weights.iterrows():
					element = element_row['element']
					atom_weight = element_row['atom_weight']
					test_df = self.reaction_df.loc[(self.reaction_df['Target'] == element)]
					

					# for rxn_string in reaction_name:
					for rxn_index, rxn_row in test_df.iterrows():
						rxn_string = cm.name+'('+self.particle+',x)'+rxn_row['Product']
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
							update_index = compound_rxn_df[compound_rxn_df['Name']==rxn_string].index.tolist()
							if self.calculate_neutron_yields:
								# print(ci.Reaction(rxn))
								# print(ci.Reaction(rxn).xs)
								if np.shape(rxn_row['XS'])[0] != np.shape(compound_rxn_df.iloc[update_index[0]]['Energy'])[0]:
									# print('Product ',ci.Reaction(rxn), ' has ', np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0], 'energy points, adding interpolated XS data with ',np.shape(ci.Reaction(rxn).xs)[0], ' more...')
									f_out = interpolate.interp1d(rxn_row['Energy'], rxn_row['XS'])
									compound_rxn_df.at[update_index[0],'XS']= compound_rxn_df.at[update_index[0],'XS'] + f_out(compound_rxn_df.iloc[update_index[0]]['Energy'])*atom_weight
								else:
									# print('Product ',ci.Reaction(rxn), ' has ', np.shape(reaction_df.iloc[update_index[0]]['Energy'])[0], 'energy points, adding interpolated XS data with ',np.shape(ci.Reaction(rxn).xs)[0], ' more...')
									compound_rxn_df.at[update_index[0],'XS']= compound_rxn_df.at[update_index[0],'XS'] + rxn_row['XS']*atom_weight
							else:
								compound_rxn_df.at[update_index[0],'XS']= compound_rxn_df.at[update_index[0],'XS'] + rxn_row['XS']*atom_weight
							compound_rxn_df.at[update_index[0],'Subtargets']= compound_rxn_df.at[update_index[0],'Subtargets'] + ',' + element


			self.compound_rxn_df = compound_rxn_df
			print(self.compound_rxn_df)

			# Export compound XS data to csv
			if self.save_csv:
				with open(self.compound_cross_sections_csv, 'w') as f:
					if bool(self.enriched_targets):
						f.write('Note: all XS calculated using target enrichment of '+str(self.enriched_targets).replace(",","")+',,,,,\n')
						self.compound_rxn_df.to_csv(f, index=False, mode='a')
					else:
						f.write('Note: all XS calculated using natural abundance targets ,,,,,\n')
						self.compound_rxn_df.to_csv(f, index=False, mode='a')
				print('XS data for all compounds in this stack have been saved to ', self.compound_cross_sections_csv, '.  Run this script again to generate activity and yield estimates.')



