import numpy as np
import pandas as pd
import re
import signal
import threading
import time


from .data import _get_connection
from .data import download
from .isotope import Isotope
from .compound import Compound
from .element import Element
from .spectrum import Spectrum


class Query(object):
	""" Search engine for decay data

	Provides methods to search through the Curie decay data library, to identify possible 
	parent isotope candidates for lines in decay spectra.

	Parameters
	----------
	mode :  str, optional
		Selects whether to search through the decay gamma or decay alpha libraries. Options are 'gammas'
		or 'alphas'. Defaults to 'gammas'

	Attributes
	-------
	result : pd.DataFrame
		Table of possible candidate decay lines, taken from the Curie decay database.

	Examples
		--------
		>>> qry = ci.Query()
		>>> print(qry.result.to_string(index=False))

		>>> qry = ci.Query(mode='alphas')

	"""

	def __init__(self, **kwargs):
		self.mode = str(kwargs['mode']) if 'mode' in kwargs else 'None'

		# Check to get the right decay library
		if self.mode.lower() == 'gammas'.lower():
			# Decay gammas
			self.mode = 'gammas'
		elif self.mode.lower() == 'alphas'.lower():
			# Decay alphas
			self.mode = 'alphas'
		elif self.mode == 'None':
			self.mode = 'gammas'
		else:
			print('Unsupported decay radiation type \"'+str(self.mode)+'\" selected.')
			print('Valid options are currently limited to \"gammas\", \"alphas\".')
			quit()


		# Grab half-life db
		ISOTOPE_LIST = np.asarray(list(map(str, pd.read_sql('SELECT name FROM chart', _get_connection('decay'))['name'])),dtype=str)
		A_LIST = np.asarray(list(map(str, pd.read_sql('SELECT A FROM chart', _get_connection('decay'))['A'])),dtype=np.int64)
		Z_LIST = np.asarray(list(map(str, pd.read_sql('SELECT Z FROM chart', _get_connection('decay'))['Z'])),dtype=np.int64)
		N_LIST = A_LIST - Z_LIST
		HALF_LIFE_LIST = np.asarray(list(map(str, pd.read_sql('SELECT half_life FROM chart', _get_connection('decay'))['half_life'])),dtype=np.float64)

		old_df = pd.DataFrame({'name': ISOTOPE_LIST, 'A': A_LIST, 'Z': Z_LIST, 'N': N_LIST, 'half_life': HALF_LIFE_LIST})

		# Look for outdated decay database and prompt the user to patch it
		if len(ISOTOPE_LIST) < 4369:
			print('The local version of decay.db appears to be out of date.')
			print('WARNING: This will write over your current database, please back it up if you have made local changes.')
			input_response = input('Would you like to update your local version? (y/N)')
			if input_response.lower() in ["yes", "y"]:
				# bool_list.append(True)
				download('decay',True) 
				print('decay.db has been updated!')
				quit()
			else:
				print('Keeping local version of decay.db.')

		# Grab decay gamma db
		NAME_LIST = np.asarray(list(map(str, pd.read_sql('SELECT isotope FROM '+self.mode, _get_connection('decay'))['isotope'])),dtype=str)
		ISOMER_LIST = np.asarray(list(map(str, pd.read_sql('SELECT isomer FROM '+self.mode, _get_connection('decay'))['isomer'])),dtype=str)
		ENERGY_LIST = np.asarray(list(map(str, pd.read_sql('SELECT energy FROM '+self.mode, _get_connection('decay'))['energy'])),dtype=np.float64)
		INTENSITY_LIST = np.asarray(list(map(str, pd.read_sql('SELECT intensity FROM '+self.mode, _get_connection('decay'))['intensity'])),dtype=np.float64)
		UNC_INTENSITY_LIST = np.asarray(list(map(str, pd.read_sql('SELECT unc_intensity FROM '+self.mode, _get_connection('decay'))['unc_intensity'])),dtype=np.float64)
		DECAY_MODE_LIST = np.asarray(list(map(str, pd.read_sql('SELECT decay_mode FROM '+self.mode, _get_connection('decay'))['decay_mode'])),dtype=str)

		new_list = [name+isomer for name,isomer in zip(NAME_LIST,ISOMER_LIST)]

		# Find unique isotopes
		unique_isotopes = set(new_list)
		# Form pandas database from dictionary of lists 
		df = pd.DataFrame({'name': new_list, 'isotope': NAME_LIST, 'isomer': ISOMER_LIST, 'energy': ENERGY_LIST, 'intensity': INTENSITY_LIST, 'unc_intensity': UNC_INTENSITY_LIST, 'decay_mode':DECAY_MODE_LIST} )
		
		# Merge into one db
		result = pd.merge(df,old_df,how="left",on=["name"])#.convert_dtypes()#.astype({'energy': 'float64','intensity': 'float64','unc_intensity': 'float64','A': 'Int64','Z': 'Int64','half_life': 'float64'}).convert_dtypes()

		# Drop unneeded columns
		self.result = result.drop(columns=['isotope', 'isomer'])



	def bg_thread(self,ip, spectrum, exit_event):
		spectrum.isotopes = [str(ip.name)]
		for i in range(1, 10):
			time.sleep(0.1)  # do some work...

			if exit_event.is_set():
				break

	def signal_handler(self,signum, frame):
		print()
		raise KeyboardInterrupt
		exit_event.set()

	def half_life(self,data, units='s', unc=False):
		half_conv = {'ns':1E-9, 'us':1E-6, 'ms':1E-3, 'sec':1.0,
					's':1.0, 'm':60.0, 'min':60.0, 'h':3600.0, 'hr':3600.0,
					'd':86400.0, 'y':31557.6E3, 'yr':31557.6E3,  'ky':31557.6E6,
					'My':31557.6E9, 'Gy':31557.6E12}

		if np.size(data) == 1:
			try:
				return data/half_conv[units]
			except TypeError:
				return data/[half_conv[x] for x in units]
		else:
			return data/[half_conv[x] for x in units]


	def optimum_units(self,array):
		units_array = np.empty(np.size(array), dtype=object)
		i=0
		for data in array:
			for units in ['ns','us','ms','s','m','h','d','y']:
				if self.half_life(data, units)>1.0:
					units_array[i] = units
			i=i+1

		optimum_half_lives = self.half_life(array,units=units_array)
		return ["%.2f " % value+str(unit) for unit,value in zip(units_array,optimum_half_lives)]

	def parse_particle(self,particle):
		### Return Z, A, N of incident particle
		try:
			# Check for basic particles
			if particle in ['p', 'd', 't', 'a', 'n']:
				return {'p':(1, 1, 0), 'd':(1, 2, 1), 't':(1, 3, 2), 'a':(2, 4, 2), 'n':(0, 1, 1)}[particle]

			# Check if Curie compound
			if isinstance(particle, Compound):
				elements = list(map(str, particle.elements))

				min_A_isotope = [int(min(row)) for row in [[Isotope(x).A for x in Element(el).isotopes] for el in elements]]
				max_A_isotope = [int(max(row)) for row in [[Isotope(x).A for x in Element(el).isotopes] for el in elements]]

				min_Z_isotope = [int(min(row)) for row in [[Isotope(x).Z for x in Element(el).isotopes] for el in elements]]
				max_Z_isotope = [int(max(row)) for row in [[Isotope(x).Z for x in Element(el).isotopes] for el in elements]]

				min_N_isotope = [int(min(row)) for row in [[Isotope(x).N for x in Element(el).isotopes] for el in elements]]
				max_N_isotope = [int(max(row)) for row in [[Isotope(x).N for x in Element(el).isotopes] for el in elements]]

				return max_Z_isotope, min_A_isotope,  min_N_isotope, max_A_isotope,  max_N_isotope

			### check if element or isotope
			if ''.join(re.findall('[A-Z]+', particle.upper())).title()==particle.title() or particle=='P':
				if particle=='P':
					print("WARNING: Assumed particle type P (phosphorus). If proton, use beam='p' (lower case).")
				elif particle=='N':
					print("WARNING: Assumed particle type N (nitrogen). If neutron, use beam='n' (lower case).")
				el = Element(particle.title())
				min_A_isotope = int(min([Isotope(x).A] for x in el.isotopes)[0])
				max_A_isotope = int(max([Isotope(x).A] for x in el.isotopes)[0])

				return el.Z, min_A_isotope,  min_A_isotope-el.Z, max_A_isotope,  max_A_isotope-el.Z

			else:
				ip = Isotope(particle)
				return ip.Z, ip.A, ip.N
		except ValueError:
			return False
			# print('Beam/Target \''+particle+'\' not parsed correctly.')
		except KeyError:
			return False
			# print('Beam/Target \''+particle+'\' not parsed correctly.')

	def decay_search(self,sort_by='intensity', energy=[],min_half_life='',max_half_life='',time_since_eob='',min_A=1,max_A=400,min_Z=1,max_Z=200,min_N=1,max_N=400,A=[],Z=[],N=[],solver = {"beam": None, "target": None, "energy": None},interactive=False,verbose=True):

		""" Searches through the results dataframe, to identidy possible candidates for gamma lines
		or alpha peaks seen in a spectrum, along with tools to narrow down that range. Does not 
		alter the Query object's dataframe, so users can run this iteratively if desired.

		Parameters
		----------

		sort_by :  str array_like or tuple, optional
			Sets the options for what quantities to sort the output dataframe by, using 
			pandas.DataFrame.sort_values(). Options are 'intensity', 'energy', 'A', 'Z', 'N', 'half_life', 
			'decay_mode', 'name'. If an array of strings is passed, the output will be non-independetly 
			sorted - the results will be first sorted by sort_by[0], and then by sort_by[1] within each 
			sort_by[0], and so on.  Defaults to sorting by 'intensity'.

			If a tuple is passed instead, users can manually specify whether the sorting occurs in 
			ascending or descending order. The tuple must be formatted such as 
			sort_by=('Z',False, 'A',True, 'energy',False), consisting of alternating string/boolean pairs 
			of the above keywords, along with a boolean representing ascending=bool for that keyword.

		energy :  array_like, optional
			Sets the range for filtering decay radiation based on an energy window, in keV. A 1-length
			array searches within a window of energy +/- 1.5 keV for gammas, or energy +/- 5 keV for alphas.
			A 2-length array searches within a window of energy[0] +/- energy[1]. Defaults to returning a 
			dataframe unfiltered on energy.

		min_half_life : str, optional
			Lower bound for filtering on the half-life of the parent radionuclide. Valid units for 
			half-life are []'ns','us','ms','s','m','h','d','y'].  Defaults to '0 s', returning a 
			dataframe unfiltered on half-life.

		max_half_life : str, optional
			Upper bound for filtering on the half-life of the parent radionuclide. Valid units for 
			half-life are []'ns','us','ms','s','m','h','d','y'].  Defaults to '0 s', returning a 
			dataframe unfiltered on half-life.

		time_since_eob : str, optional
			Helper parameter, to filter out all isotopes which will have decayed out by the time a spectrum 
			is collected, assumed to be 10 half-lives since the end of an irradiation, by setting 
			min_half_life = 10 * time_since_eob.  Defaults to '0 s', returning a dataframe unfiltered 
			on half-life.

		min_A : int or str, optional
			Filters out all radionuclides with A < min_A. If min_A is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the A of the specified isotope, or the A of 
			the lightest stable isotope (if an element is passed). If the solver parameter is passed, 
			the min_A set by it takes priority over manually specifying min_A. Defaults to 1.

		max_A : int or str, optional
			Filters out all radionuclides with A > max_A. If max_A is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the A of the specified isotope, or the A of 
			the heaviest stable isotope (if an element is passed). If the solver parameter is passed, 
			the max_A set by it takes priority over manually specifying max_A. Defaults to 400.

		min_Z : int or str, optional
			Filters out all radionuclides with Z < min_Z. If min_Z is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the Z of the specified isotope or element.  
			Defaults to 1.

		max_Z : int or str, optional
			Filters out all radionuclides with Z > max_Z. If max_Z is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the Z of the specified isotope or element. 
			If the solver parameter is passed, the max_Z set by it takes priority over manually 
			specifying max_Z. Defaults to 200.

		min_N : int or str, optional
			Filters out all radionuclides with N < min_N. If min_N is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the N of the specified isotope, or the N of 
			the lightest stable isotope (if an element is passed). Defaults to 1.

		max_N : int or str, optional
			Filters out all radionuclides with N > max_N. If max_N is passed as a str (followig Curie 
			notation for isotopes or elements), it chooses the N of the specified isotope, or the N of 
			the heaviest stable isotope (if an element is passed). If the solver parameter is passed, 
			the max_N set by it takes priority over manually specifying max_N. Defaults to 400.

		A : int, optional
			Returns only radionuclides with the specified A. Defaults to Any.

		Z : int, optional
			Returns only radionuclides with the specified Z. Defaults to Any.

		N : int, optional
			Returns only radionuclides with the specified N. Defaults to Any.

		solver : dict, optional
			Physics-guided helper parameter to narrow down a list of expected possible reaction products for
			an activation experiment. Dicts must be formatted as {"beam": particle, "target": target, "energy": energy}, 
			and an empty dict will default to solver = {}, disabling this functionality. In this mode, the 
			maximum posisble compound nucleus for a given beam+target combination is determined, and all
			possible reaction products (assuming an average 5 MeV/nucleon separation energy) are listed for
			this compound nuclear system. The lower range of this set is intentionally set to be slightly wider
			than pure Q-value based estimates, to account for trace impurities of other light elements in the
			target system. For the above dict structure, the following values are allowed:

			particle : str - any of the basic Curie light ions ('p', 'd', 't', 'a'), neutrons ('n'), or
			any Curie-formatted string notation for isotopes or elements.

			target : str - any Curie-formatted string notation for isotopes or elements, or a ci.Compound
			object.  For the case of elements passed as a target, the maximum possible compound nucleus is 
			determined based on the heaviest stable isotope of that element. In the case of passing a ci.Compound, 
			solver adds together the individual energetically-possible list of candidate isotopes for each 
			element present in the compound.

			energy : int or float - beam energy, in MeV.


		interactive : boolean or tuple, optional
			Enables a semi-interactive mode, to eliminate possible candidates based on whether or not their
			other intense decay lines are visible in the spectrum. This mode is best used to further narrow 
			down an initial list of candidate radionuclides using a combination of the raneg of other filter
			parameters available. This mode should not be used blindly as a black-box, but rather to help 
			the expert user narrow down possible candidates. Defaults to False.

			If a boolean is passed, interactive=True enables this mode, with user-prompted questions to 
			look for intense gammas to remove false candidates.

			A tuple (of length 1 or 2) can be instead passed, to supply a spectrum filename. In this mode, 
			rather than relying upon user-prompted questions, Curie will iteratively fit the supplied 
			spectrum for each candidate isotope, removing all of those which have no visible gammas. For 
			those isotopes which do have visible lines, the candidate with the best average chi2 for all 
			of its fitted peaks is suggested as the most likely candidate. interactive[0] must be a str
			containing the filename to a spectrum. interactive[1] is optional, and may be a str
			containing the filename of a curie calibration .json file. If no calibration file is passed, 
			the spectrum's MCA calibration will be used by default. 

		verbose : boolean, optional
			Selector to print the results table to the terminal when running.  Defaults to True.

		Returns
		-------
		result : pd.DataFrame
			Table of possible candidate decay lines, taken from the Curie decay database.

		

		Examples
			--------
			>>> qry.decay_search(sort_by=['half_life','intensity'])

			>>> qry.decay_search(sort_by='Z',min_A='Ca',max_Z='Tb')

			>>> qry.decay_search(energy=[846.2,2],sort_by='half_life',time_since_eob='1 m',solver = {"beam": 'd', "target": 'Fe', "energy": 40})

			>>> qry.decay_search(energy=[846.2,10],sort_by='Z',solver = {"beam": 'd', "target":  ci.Compound('Brass', weights={'Zn':-33,'Cu':-66}), "energy": 40}, min_A=30)
			
			>>> qry.decay_search(energy=[645.0,1],sort_by='intensity',time_since_eob='12 h',solver = {"beam": 'd', "target": 'TM', "energy": 40}, min_A=10)#,min_A='Ca',max_Z='Tb')#,max_N=130,min_N=50,max_A=100, min_A=5, min_Z=4, max_Z=90)
			
			>>> qry.decay_search(sort_by=['A','energy'], min_Z='Ho',max_Z='Ho')

			>>> qry.decay_search(energy=[628.2,100],sort_by=('Z',False,'A',True,'energy',False))

			>>> qry.decay_search(sort_by='half_life', energy=[290.0,2] , min_half_life='3 h',solver = {"beam": 'p', "target": 'Tl', "energy": 50},interactive=True)

			>>> qry.decay_search(sort_by='half_life', energy=[290.0,1] , min_half_life='3 h',solver = {"beam": 'p', "target": 'Tl', "energy": 50}, interactive='Spectrum.Spe')

			>>> qry.decay_search(sort_by='half_life', energy=[645.0,2],solver = {"beam": 'd', "target": 'Tm', "energy": 20}, interactive=('Spectrum.Chn','Calibration.json'))
			
			>>> df = qry.decay_search(verbose=False)
			>>> print(df.to_string(index=False))


		"""

		# Make local copy, to avoid altering the Query object state, in case you want to run decay_search() multiple times on one Query object
		result = self.result


		# Filter on gamma energy range
		if not all([type(x) in (int, float) for x in energy]):
			print('\'energy\' must be a numeric array_like of length 1 or 2.')
			return True

		if np.size(energy) == 1:
			energy=np.array(energy,dtype=float)
			if mode=='gammas':
				energy_bound = 1.5
			elif mode=='alphas': 
				energy_bound = 5.0
			result = result[(result.energy.ge(energy[0]-energy_bound)) & (result.energy.le(energy[0]+energy_bound))]
		elif np.size(energy) == 2:
			energy=np.array(energy)
			energy_bound = energy[1]
			result = result[(result.energy.ge(energy[0]-energy_bound)) & (result.energy.le(energy[0]+energy_bound))]
		elif np.size(energy) > 2:
			print('\'energy\' must be a numeric array_like of length 1 or 2.')
			return True

		
		# Use beam info to smart-select possible products
		if any(solver.values()):
			beam_info = self.parse_particle(solver["beam"])
			if np.size(beam_info) == 3:
				A_beam = beam_info[1]
				Z_beam = beam_info[0]
				N_beam = beam_info[2]
			elif np.size(beam_info) == 5:
				A_beam = beam_info[3]
				Z_beam = beam_info[0]
				N_beam = beam_info[4]
			else:
				print('Beam \''+solver["beam"]+'\' not properly parsed')

			target_info = self.parse_particle(solver["target"])
			if np.size(target_info) == 3:
				# single isotope
				A_target = target_info[1]
				Z_target = target_info[0]
				N_target = target_info[2]
			elif np.size(target_info) == 5:
				# single element
				A_target = target_info[3]
				min_A_target = target_info[1]
				Z_target = target_info[0]
				N_target = target_info[4]
			elif np.size(target_info) > 5:
				# compound
				A_target = target_info[3]
				min_A_target = target_info[1]
				Z_target = target_info[0]
				N_target = target_info[4]
			else:
				print('Target \''+solver["target"]+'\' not properly parsed')

			if np.size(A_target) > 1:
				compound_A = [x+A_beam for x in A_target]
				compound_Z = [x+Z_beam for x in Z_target]
				compound_N = [x+N_beam for x in N_target]
			else:
				compound_A = A_target+A_beam
				compound_Z = Z_target+Z_beam
				compound_N = N_target+N_beam

			# Removed extra padding above max compound nucleus for trace impurities. May add back in if needed.
			max_A = compound_A# +1
			max_Z = compound_Z# +1
			max_N = compound_N# +1



			# Assumes 5 MeV/nucleon average separation energy, with some padding
			beam_energy = solver["energy"]
			if beam_energy > 100:
				nucleons_removed = int(beam_energy / 5)
			elif beam_energy > 50:
				nucleons_removed = int(beam_energy / 5)+5
			else:
				nucleons_removed = int(beam_energy / 5)+10
			if np.size(A_target) > 1:
				# min_A = [x-nucleons_removed for x in min_A_target]
				min_A = np.max(np.array([[x-nucleons_removed for x in min_A_target],np.full_like(min_A_target, min_A)]),axis=0)
			else:
				min_A = np.max([min_A_target - nucleons_removed,min_A])



		# Filter on half-life
		half_conv = {'ns':1E-9, 'us':1E-6, 'ms':1E-3, 'sec':1.0,
					's':1.0, 'm':60.0, 'min':60.0, 'h':3600.0, 'hr':3600.0,
					'd':86400.0, 'y':31557.6E3, 'yr':31557.6E3,  'ky':31557.6E6,
					'My':31557.6E9, 'Gy':31557.6E12}
		if not all([type(min_half_life) in (int, float)]):
			min_half_life = min_half_life.replace(' ','')
			min_letters = ''.join(re.findall('([a-zA-Z])', min_half_life))
			min_numbers = ''.join(re.findall('([0-9.])', min_half_life))
		else:
			min_numbers = float(min_half_life)
			min_letters = ''

		if not all([type(max_half_life) in (int, float)]):
			max_half_life = max_half_life.replace(' ','')
			max_letters = ''.join(re.findall('([a-zA-Z])', max_half_life))
			max_numbers = ''.join(re.findall('([0-9.])', max_half_life))
		else:
			max_numbers = float(max_half_life)
			max_letters = ''

		if not all([type(time_since_eob) in (int, float)]):
			eob_half_life = time_since_eob.replace(' ','')
			eob_letters = ''.join(re.findall('([a-zA-Z])', eob_half_life))
			eob_numbers = ''.join(re.findall('([0-9.])', eob_half_life))
		else:
			eob_numbers = float(time_since_eob)
			eob_letters = ''
		
		try:
			if eob_numbers or isinstance(eob_numbers, float):
				if not eob_letters:
					eob_letters= 's'
				result = result[result.half_life.ge(10*float(eob_numbers)*half_conv[eob_letters])]
		except KeyError:
			print('Half-life units \''+eob_letters+'\' not parsed, available options are:',list(half_conv.keys()))
		try:
			if min_numbers or isinstance(min_numbers, float):
				if not min_letters:
					min_letters= 's'
				result = result[result.half_life.ge(float(min_numbers)*half_conv[min_letters])]
		except KeyError:
			print('Half-life units \''+min_letters+'\' not parsed, available options are:',list(half_conv.keys()))
		try:
			if max_numbers or isinstance(max_numbers, float):
				if not max_letters:
					max_letters= 's'
				result = result[result.half_life.le(float(max_numbers)*half_conv[max_letters])]
		except KeyError:
			print('Half-life units \''+max_letters+'\' not parsed, available options are:',list(half_conv.keys()))

		

		# Filter on A range
		if not ('A_target' in locals()):
			A_target = [-1]
		if (np.size(A_target) > 1):
			result_compound = result.iloc[:0]

			for i_maxA, i_Z, i_N, i_minA in zip(max_A, max_Z,max_N, min_A):
				result_loop = result[result.A.ge(i_minA)]
				result_loop = result_loop[result_loop.A.le(i_maxA)]
				result_loop = result_loop[result_loop.Z.le(i_Z)]
				result_loop = result_loop[result_loop.Z.ge(min_Z)]
				result_loop = result_loop[result_loop.N.le(i_N)]
				result_loop = result_loop[result_loop.N.ge(min_N)]
				result_compound = pd.concat([result_compound, result_loop])
			result = result_compound.drop_duplicates()
		else:
			if isinstance(min_A, str):
				try:
					min_A = int(min([Isotope(x).A] for x in Element(min_A).isotopes)[0])
				except ValueError:
					min_A = int(Isotope(min_A).A)
			if isinstance(max_A, str):
				try:
					max_A = int(max([Isotope(x).A] for x in Element(max_A).isotopes)[0])
				except ValueError:
					max_A = int(Isotope(max_A).A)
			min_A = int(min_A)
			max_A = int(max_A)
			result = result[result.A.ge(min_A)]
			result = result[result.A.le(max_A)]

			if np.size(A) == 1:
				result = result[result.A.eq(int(A))]

			# Filter on Z range
			if isinstance(min_Z, str):
				min_Z = Element(min_Z).Z
			if isinstance(max_Z, str):
				max_Z = Element(max_Z).Z
			min_Z = int(min_Z)
			max_Z = int(max_Z)
			result = result[result.Z.ge(min_Z)]
			result = result[result.Z.le(max_Z)]

			if np.size(Z) == 1:
				result = result[result.Z.eq(int(Z))]

			# Filter on N range
			if isinstance(min_N, str):
				try:
					min_N = int(min([Isotope(x).N] for x in Element(min_N).isotopes)[0])
				except ValueError:
					min_N = int(Isotope(min_N).N)
			if isinstance(max_N, str):
				try:
					max_N = int(max([Isotope(x).N] for x in Element(max_N).isotopes)[0])
				except ValueError:
					max_N = int(Isotope(max_N).N)
			min_N = int(min_N)
			max_N = int(max_N)
			result = result[result.N.ge(min_N)]
			result = result[result.N.le(max_N)]

			if np.size(N) == 1:
				result = result[result.N.eq(int(N))]

		

		# Sort output
		if isinstance(sort_by, tuple):
			# User-specified sort orders
			if sort_by[1] in [True, False]:
				if np.size(sort_by) > 2:
					result = result.sort_values(list(sort_by[0::2]),ascending=list(sort_by[1::2]))
				elif sort_by[0] == 'intensity':
					result = result.sort_values(sort_by[0],ascending=sort_by[1])
				elif sort_by[0] == 'energy':
					result = result.sort_values(sort_by[0],ascending=sort_by[1])
				elif sort_by[0] == 'A':
					result = result.sort_values([sort_by[0],'intensity'],ascending=[sort_by[1],False])
				elif sort_by[0] == 'Z':
					result = result.sort_values([sort_by[0],'intensity'],ascending=[sort_by[1],False])
				elif sort_by[0] == 'N':
					result = result.sort_values([sort_by[0],'intensity'],ascending=[sort_by[1],False])
				elif sort_by[0] == 'half_life':
					result = result.sort_values(sort_by[0],ascending=sort_by[1])
				elif sort_by[0] == 'decay_mode':
					result = result.sort_values(sort_by[0],ascending=sort_by[1])
				elif sort_by[0] == 'name':
					result = result.sort_values(sort_by[0],ascending=sort_by[1])
				else:
					print('Invalid sorting option \''+str(sort_by)+'\', sorting by intensity instead.')
		else:
			if sort_by == 'intensity':
				result = result.sort_values(sort_by,ascending=False)
			elif sort_by == 'energy':
				result = result.sort_values(sort_by,ascending=True)
			elif sort_by == 'A':
				result = result.sort_values([sort_by,'intensity'],ascending=[False,False])
			elif sort_by == 'Z':
				result = result.sort_values([sort_by,'intensity'],ascending=[False,False])
			elif sort_by == 'N':
				result = result.sort_values([sort_by,'intensity'],ascending=[False,False])
			elif sort_by == 'half_life':
				result = result.sort_values(sort_by,ascending=False)
			elif sort_by == 'decay_mode':
				result = result.sort_values(sort_by,ascending=True)
			elif sort_by == 'name':
				result = result.sort_values(sort_by,ascending=True)
			elif np.size(sort_by) > 1:
				result = result.sort_values(sort_by,ascending=False)
			else:
				print('Invalid sorting option \''+str(sort_by)+'\', sorting by intensity instead.')


		
		# Grab half life data, and convert to "pretty" units
		half_lives_array = result.half_life.to_numpy()

		# Append to results dataframe
		result["Half-Life"] = self.optimum_units(half_lives_array)


		# Cleanup formatting for 'pretty' output
		result["energy"] = result["energy"].astype(str)
		result["Intensity (%)"] = result["intensity"].astype(str)# + result["unc_intensity"].astype(str)
		result["Isotope"] = result["name"].astype(str).str.replace('g',' g').str.replace('m',' m')
		result["decay_mode"] = result["decay_mode"].astype(str).str.replace('A',' Alpha').str.replace('B',' Beta-').str.replace('E',' Beta+/E.C.').str.replace('IT',' Isomeric Transition')

		result = result.rename(columns={"energy": "Energy (keV)","decay_mode": "Decay Mode"}).drop(columns=['half_life',"name",'intensity','unc_intensity']).reindex(columns=['Energy (keV)','Intensity (%)','Half-Life','Isotope','A','Z','N','Decay Mode'])



		# Print output
		if verbose:
			if len(result.index) > 300:
				if interactive:
					print('Initial Results:')
				print(len(result.index),  self.mode ,'matched your criteria, consider narrowing your search criteria.')
				print('Truncating to the first 300 results:')
				print(result.head(300).to_string(index=False))
			elif len(result.index) == 1:
				if interactive:
					print('Initial Results:')
				print(len(result.index), str(self.mode).replace('s',''), 'matched your criteria:')
				print(result.to_string(index=False))
			elif len(result.index) == 0:
				print('No matching', self.mode ,'found!')
				return False
			else:
				if interactive:
					print('Initial Results:')
				print(len(result.index),  self.mode ,'matched your criteria:')
				print(result.to_string(index=False))

		
		# Guided exploration to eliminate candidates
		if (interactive and isinstance(interactive, bool)) or isinstance(interactive, str) or isinstance(interactive, tuple):
			if isinstance(interactive, str) or isinstance(interactive, tuple):
				candidates = sorted(list(set(result.Isotope.tolist())))
				best_chi2 = 1.0E10
				best_candidate = ''
				try:
					print('Press Ctrl-c at any point to end interactive mode.\n')
					if (isinstance(interactive, tuple) and (not isinstance(interactive[1], str))):
						print('Using calibration data from spectrum header...')
					elif (not isinstance(interactive, tuple)):
						print('Using calibration data from spectrum header...')
					for itp in candidates:
						# Look for peaks in spectrum
						if isinstance(interactive, tuple):
							sp = Spectrum(interactive[0])
							if isinstance(interactive[1], str):
								sp.cb = interactive[1]

						else:
							sp = Spectrum(interactive)

						pks = []
						ip = Isotope(itp.replace(' ',''))
						gamma_list = ip.gammas().sort_values('intensity',ascending=False,ignore_index=True)
						j = 0
						bool_list = []
						# sp_loop = sp
						print('Looking for',str(ip.name))

						# Patch to work around ci.Spectrum not handling KeyboardInterrupt exceptions
						exit_event = threading.Event()

						signal.signal(signal.SIGINT, self.signal_handler)
						th = threading.Thread(target=self.bg_thread(ip,sp,exit_event))
						th.start()
						th.join()


						# Look for peaks in spectrum
						try:
							if (isinstance(interactive, tuple) and (not isinstance(interactive[1], str))):
								sp.auto_calibrate()
							elif (not isinstance(interactive, tuple)):
								sp.auto_calibrate()
						except KeyError:
							pass
						# Show Curie spectrum plots if desired
						try:
							if isinstance(interactive, tuple):
								if len(interactive) == 2:
									if interactive[1] == False:
										pass 
									else:
										sp.plot()
								elif len(interactive) > 2:
									if interactive[2] == False:
										pass	
									else:
										sp.plot()							
								else:
									sp.plot()
							else:
								sp.plot()



							pks = sp.peaks

							# Remove candidates...
							if pks is None:
								# ...if isotope is not seen in spectrum
								print('Candidate',ip.name,'does not appear to have any visible gammas. Dropping...')
								result = result[result.Isotope != str(ip.name).replace('g', ' g').replace('m',' m')]
							else:
								candidate_gamma_loop = result[result.Isotope == itp]['Energy (keV)'].to_numpy(dtype=float)[0]
								matching_gamma = pks[(pks.energy.ge(candidate_gamma_loop-energy_bound)) & (pks.energy.le(candidate_gamma_loop+energy_bound))]
								if len(matching_gamma.index) == 0:
									# ...if no matching peaks are found
									print('The spectrum does not appear to have any visible gammas near',candidate_gamma_loop,'keV. Dropping...')
									result = result[result.Isotope != str(ip.name).replace('g', ' g').replace('m',' m')]
								elif len(matching_gamma.index) >= 1:	
									# ...unless isotope is seen in spectrum
									decay_rate_loop = pks.unc_decay_rate.to_numpy(dtype=float)
									unc_decay_rate_loop = pks.decay_rate.to_numpy(dtype=float)
									chi2_loop = pks.chi2.to_numpy(dtype=float)
									# print(candidate_gamma_loop,'keV decay rate: {:.2E}'.format(matching_gamma['decay_rate'].to_numpy(dtype=float)[0]))
									# print('Average decay rate for ',ip.name,": {:.2E}".format(np.average(decay_rate_loop)))
									print('Average chi2 for ',ip.name,": {:.2f}".format(np.average(chi2_loop)))
									if np.average(chi2_loop) < best_chi2:
										# Maintain current best candidate for later
										best_chi2 = np.average(chi2_loop)
										best_candidate = ip.name
						
						except KeyError:
							print('WARNING: Unable to fit spectrum',sp.filename, 'for candidate',ip.name+'. Dropping...')
							result = result[result.Isotope != str(ip.name).replace('g', ' g').replace('m',' m')]
							pass
						except KeyboardInterrupt:
							break
						print('--------------------------')

				except KeyboardInterrupt:
					print()
					pass
						
			
				
					

			
			if (interactive and isinstance(interactive, bool)):
				# Look for peaks with user input
				print('Press Ctrl-c at any point to end interactive mode.\n')
				candidates = list(set(result.Isotope.tolist()))
				try:
					for itp in candidates:
						ip = Isotope(itp.replace(' ',''))
						gamma_list = ip.gammas().sort_values('intensity',ascending=False,ignore_index=True)
						j = 0
						bool_list = []
						print('Looking for',str(itp).replace(' ',''))
						for i in np.arange(3):
							if (np.size(energy) == 2) & (gamma_list.loc[i,'energy'] >= (energy[0]-energy[1]) ) & (gamma_list.loc[i,'energy'] <= (energy[0]+energy[1]) ):
								print('The',str(gamma_list.loc[i,'energy']),'keV peak from',ip, 'falls within the energy search window of',energy[0],'+/-',energy[1],'keV. Skipping....')
								j=j+1
							while True & (j < len(gamma_list.index)):
								input_response = input('Is a peak near '+str(gamma_list['energy'][j])+' keV visible? (Y/N)  ')
								if input_response.lower() in ["yes", "y"]:
									bool_list.append(True)
									break
								elif input_response.lower() in ["no", "n"]:
									bool_list.append(False)
									break
								else:
									print('Response not parsed properly, please try again.')
							j=j+1

						if all(bool_list):
							print(ip,'appears to be a strong candidate!')
						if not any(bool_list):
							print('Candidate',ip,'does not appear to have any visible gammas. Dropping...')
							result = result[result.Isotope != str(ip).replace('g', ' g').replace('m',' m')]
						print('--------------------------')
				except KeyboardInterrupt:
					print()
					pass

			# Print output
			if verbose:
				print()
				print('Interactively-Filtered Results:')
				if len(result.index) > 300:
					print(len(result.index), self.mode ,'matched your criteria, consider narrowing your search criteria.')
					print('Truncating to the first 300 results:')
					print(result.head(3000).to_string(index=False))
				elif len(result.index) == 1:
					print(len(result.index),  str(self.mode).replace('s','') ,'matched your criteria:')
					print(result.to_string(index=False))
				elif len(result.index) == 0:
					print('No matching', self.mode ,'found!')
				else:
					print(len(result.index),  self.mode ,'matched your criteria:')
					print(result.to_string(index=False))
				if 'best_chi2' in locals():
					if best_chi2 < 1e9:
						print('Candidate',best_candidate, 'is the most likely candidate, with an average chi2 of {:.2f}.'.format(best_chi2))




		return result
