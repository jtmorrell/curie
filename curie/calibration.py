
import numpy as np
import pandas as pd
import json

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .isotope import Isotope
from .data import _get_connection
from .plotting import _init_plot, _draw_plot, colormap
from ._log import _get_logger
from ._diagnostics import _diagnostics_frame, _at_bound, _unmoved

_log = _get_logger('calibration')

# storage groups of _calib_data: the classic groups hold the points the fit
# used (raw readers keep seeing used-points-only); the *_dropped siblings hold
# rejected points with the reason they were removed
_CALIB_GROUPS = ['engcal', 'rescal', 'effcal',
				 'engcal_dropped', 'rescal_dropped', 'effcal_dropped']




_MU_Be = [[1.000000e+00, 1.117770e+03],
		 [1.500000e+00, 3.324450e+02],
		 [2.000000e+00, 1.381950e+02],
		 [3.000000e+00, 3.934950e+01],
		 [4.000000e+00, 1.606725e+01],
		 [5.000000e+00, 8.082650e+00],
		 [6.000000e+00, 4.676800e+00],
		 [8.000000e+00, 2.079400e+00],
		 [1.000000e+01, 1.196210e+00],
		 [1.500000e+01, 5.679500e-01],
		 [2.000000e+01, 4.164350e-01],
		 [3.000000e+01, 3.315200e-01],
		 [4.000000e+01, 3.034000e-01],
		 [5.000000e+01, 2.874900e-01],
		 [6.000000e+01, 2.762050e-01],
		 [8.000000e+01, 2.591850e-01],
		 [1.000000e+02, 2.456800e-01],
		 [1.500000e+02, 2.201500e-01],
		 [2.000000e+02, 2.014650e-01],
		 [3.000000e+02, 1.750655e-01],
		 [4.000000e+02, 1.567320e-01],
		 [5.000000e+02, 1.431715e-01],
		 [6.000000e+02, 1.323675e-01],
		 [8.000000e+02, 1.162910e-01],
		 [1.000000e+03, 1.045620e-01],
		 [1.022000e+03, 1.034335e-01],
		 [1.250000e+03, 9.349900e-02],
		 [1.500000e+03, 8.506300e-02],
		 [2.000000e+03, 7.285300e-02],
		 [2.044000e+03, 7.198350e-02],
		 [3.000000e+03, 5.805300e-02],
		 [4.000000e+03, 4.928400e-02],
		 [5.000000e+03, 4.341950e-02],
		 [6.000000e+03, 3.923850e-02],
		 [7.000000e+03, 3.609350e-02],
		 [8.000000e+03, 3.365150e-02],
		 [9.000000e+03, 3.170900e-02],
		 [1.000000e+04, 3.009950e-02]]


_MU_Ge = [[1.1100000e+01, 1.0465018e+03, 4.0529322e-01, 1.0550186e+03],
		 [1.5000000e+01, 4.8119920e+02, 4.8167827e-01, 4.8694804e+02],
		 [2.0000000e+01, 2.2058512e+02, 5.4667210e-01, 2.2473706e+02],
		 [3.0000000e+01, 7.1062050e+01, 6.1800030e-01, 7.3723550e+01],
		 [4.0000000e+01, 3.1102289e+01, 6.5100290e-01, 3.3034538e+01],
		 [5.0000000e+01, 1.6203212e+01, 6.6431040e-01, 1.7752205e+01],
		 [6.0000000e+01, 9.4483250e+00, 6.6750420e-01, 1.0768429e+01],
		 [8.0000000e+01, 4.0002345e+00, 6.5951970e-01, 5.0573823e+00],
		 [1.0000000e+02, 2.0413705e+00, 6.4408300e-01, 2.9542650e+00],
		 [1.5000000e+02, 5.9883750e-01, 5.9883750e-01, 1.3259593e+00],
		 [2.0000000e+02, 2.5188436e-01, 5.5785040e-01, 8.8415030e-01],
		 [3.0000000e+02, 7.6491510e-02, 4.9136613e-01, 6.0203130e-01],
		 [4.0000000e+02, 3.4131076e-02, 4.4282037e-01, 4.9647621e-01],
		 [5.0000000e+02, 1.8848743e-02, 4.0561260e-01, 4.3712476e-01],
		 [6.0000000e+02, 1.1912874e-02, 3.7596349e-01, 3.9666996e-01],
		 [8.0000000e+02, 6.0895120e-03, 3.3098414e-01, 3.4205598e-01],
		 [1.0000000e+03, 3.7899760e-03, 2.9787508e-01, 3.0484821e-01],
		 [1.0220000e+03, 3.6015418e-03, 2.9473451e-01, 3.0138826e-01],
		 [1.2500000e+03, 2.4336756e-03, 2.6657584e-01, 2.7152623e-01],
		 [1.5000000e+03, 1.7517993e-03, 2.4235619e-01, 2.4789211e-01],
		 [2.0000000e+03, 1.0757783e-03, 2.0690501e-01, 2.1749778e-01],
		 [2.0440000e+03, 1.0390496e-03, 2.0434997e-01, 2.1552827e-01],
		 [3.0000000e+03, 5.8020700e-04, 1.6288380e-01, 1.8758252e-01],
		 [4.0000000e+03, 3.8905807e-04, 1.3584296e-01, 1.7432825e-01],
		 [5.0000000e+03, 2.9031642e-04, 1.1731892e-01, 1.6810034e-01],
		 [6.0000000e+03, 2.3069882e-04, 1.0369204e-01, 1.6538561e-01],
		 [7.0000000e+03, 1.9093601e-04, 9.3258960e-02, 1.6474685e-01],
		 [8.0000000e+03, 1.6267088e-04, 8.4848620e-02, 1.6522592e-01],
		 [9.0000000e+03, 1.4164503e-04, 7.8035180e-02, 1.6639698e-01],
		 [1.0000000e+04, 1.2535665e-04, 7.2286340e-02, 1.6799388e-01]]

_MU_Be, _MU_Ge = np.array(_MU_Be), np.array(_MU_Ge)
_MU_W = lambda E_keV: np.exp(interp1d(np.log(_MU_Be[:,0]), np.log(_MU_Be[:,1]), bounds_error=False, fill_value='extrapolate', kind='quadratic')(np.log(E_keV)))
_TAU = lambda E_keV: np.exp(interp1d(np.log(_MU_Ge[:,0]), np.log(_MU_Ge[:,1]), bounds_error=False, fill_value='extrapolate', kind='quadratic')(np.log(E_keV)))
_SIGMA = lambda E_keV: np.exp(interp1d(np.log(_MU_Ge[:,0]), np.log(_MU_Ge[:,2]), bounds_error=False, fill_value='extrapolate', kind='quadratic')(np.log(E_keV)))
_MU = lambda E_keV: np.exp(interp1d(np.log(_MU_Ge[:,0]), np.log(_MU_Ge[:,3]), bounds_error=False, fill_value='extrapolate', kind='quadratic')(np.log(E_keV)))


__doctest_skip__ = ['Calibration.calibrate', 'Calibration.saveas',
					'Calibration.plot', 'Calibration.plot_engcal',
					'Calibration.plot_rescal', 'Calibration.plot_effcal']


class Calibration(object):
	"""Calibration for HPGe spectra

	Provides methods for calculating and fitting energy, efficiency and resolution
	calibrations for HPGe spectra.  Each Spectrum class contains a Calibration, 
	which can be loaded from a file, or fit to calibration spectra from known sources.
	
	Parameters
	----------
	filename : str, optional
		Path to a .json file storing calibration data.  The .json file can be produced by
		calling `cb.saveas('example_calib.json')` after performing a calibration fit with
		the `cb.calibrate()` function.  This allows past calibrations to be recalled without
		needing to re-perform the calibration fits.

	Attributes
	----------
	engcal : np.ndarray
		Energy calibration parameters. length 2 or 3 array, depending on whether the calibration
		is linear or quadratic.

	effcal : np.ndarray
		Efficiency calibration parameters, for the semi-empirical efficiency model
		(see the `eff` method).  Length 5 array, or length 7 if the two low-energy
		attenuation terms (detector window and dead layer) are included.

	unc_effcal : np.ndarray
		Efficiency calibration covariance matrix. shape 5x5 or 7x7, depending on the length of effcal.

	rescal : np.ndarray
		Resolution calibration parameters.  length 2 array if resolution calibration is of the form
		R = a + b*chan (default), or length 1 if R = a*sqrt(chan).


	Examples
	--------
	>>> cb = Calibration()
	>>> print(cb.engcal)
	[0.  0.3]
	>>> cb.engcal = [0.1, 0.2, 0.003]
	>>> print(cb.engcal)
	[0.1   0.2   0.003]
	>>> cb.saveas('test_calib.json')
	>>> cb = Calibration('test_calib.json')
	>>> print(cb.engcal)
	[0.1   0.2   0.003]

	"""

	def __init__(self, filename=None):
		self._engcal = np.array([0.0, 0.3])
		self._effcal = np.array([1.87867638e-02, 8.82671055e+01, 2.09673703e+00, 1.66193806e+00,
								 3.90089610e-01, 3.79361265e+00, 1.24058591e-02])
		self._unc_effcal = np.array([[ 1.58657336e-05, -8.87692037e-17,  3.38230753e-04,  2.29975399e-04,
									  -2.52612161e-04,  3.15625462e-03, -3.28997933e-06],
									 [-8.87692037e-17,  4.96869464e-28, -1.88956043e-15, -1.27662324e-15,
									   1.40932100e-15, -1.76678605e-14,  1.84808443e-17],
									 [ 3.38230753e-04, -1.88956043e-15,  7.87574389e-03,  5.80822792e-03,
									  -5.65223905e-03,  6.70142358e-02, -6.81029068e-05],
									 [ 2.29975399e-04, -1.27662324e-15,  5.80822792e-03,  4.99639027e-03,
									  -4.20485186e-03,  4.51280328e-02, -4.29072666e-05],
									 [-2.52612161e-04,  1.40932100e-15, -5.65223905e-03, -4.20485186e-03,
									   4.20699447e-03, -5.00275244e-02,  5.06228293e-05],
									 [ 3.15625462e-03, -1.76678605e-14,  6.70142358e-02,  4.51280328e-02,
									  -5.00275244e-02,  6.28286136e-01, -6.57808573e-04],
									 [-3.28997933e-06,  1.84808443e-17, -6.81029068e-05, -4.29072666e-05,
									   5.06228293e-05, -6.57808573e-04,  7.23950095e-07]])
		self._rescal = np.array([2.0, 4E-4])
		self._calib_data = {}
		self._diagnostics = None

		if filename is not None:
			if filename.endswith('.json'):
				with open(filename) as f:
					js = json.load(f)
					if 'engcal' in js:
						self.engcal = js['engcal']
					if 'rescal' in js:
						self.rescal = js['rescal']
					if 'effcal' in js:
						self.effcal = js['effcal']
					if 'unc_effcal' in js:
						self.unc_effcal = js['unc_effcal']
					if '_calib_data' in js:
						for c in _CALIB_GROUPS:
							if c in js['_calib_data']:
								self._calib_data[c] = {str(i):np.array(js['_calib_data'][c][i]) for i in js['_calib_data'][c]}


	@property
	def engcal(self):
		return self._engcal

	@engcal.setter
	def engcal(self, cal):
		self._engcal = np.asarray(cal)

	@property
	def effcal(self):
		return self._effcal

	@effcal.setter
	def effcal(self, cal):
		self._effcal = np.asarray(cal)

	@property
	def unc_effcal(self):
		return self._unc_effcal

	@unc_effcal.setter
	def unc_effcal(self, cal):
		self._unc_effcal = np.asarray(cal)

	@property
	def rescal(self):
		return self._rescal

	@rescal.setter
	def rescal(self, cal):
		self._rescal = np.asarray(cal)


	def eng(self, channel, engcal=None):
		"""Energy calibration function

		Returns the calculated energy given an input array of
		channel numbers.  The `engcal` can be supplied, or
		if `engcal=None` the Calibration object's energy
		calibration is used (cb.engcal).

		Parameters
		----------
		channel : array_like
			Spectrum channel number.  The maximum channel number should
			be the length of the spectrum.

		engcal : array_like, optional
			Optional energy calibration. If a length 2 array, calibration
			will be engcal[0] + engcal[1]*channel.  If length 3, then
			engcal[0] + engcal[1]*channel + engcal[2]*channel**2

		Returns
		-------
		energy: np.ndarray
			Calibrated energy corresponding to the given channels.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.engcal)
		[0.  0.3]
		>>> print(cb.eng(np.arange(10)))
		[0.  0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7]
		>>> cb.engcal = [0.1, 0.2, 0.003]
		>>> print(cb.eng(np.arange(10)))
		[0.1   0.303 0.512 0.727 0.948 1.175 1.408 1.647 1.892 2.143]

		"""

		channel = np.asarray(channel)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==2:
			return engcal[0] + engcal[1]*channel

		elif len(engcal)==3:
			return engcal[0] + engcal[1]*channel + engcal[2]*channel**2
		
	def eff(self, energy, effcal=None):
		"""Efficiency calibration function

		Returns the calculated (absolute) efficiency given an input array of 
		energies. The `effcal` can be supplied, or if `effcal=None`, the
		calibration object's internal efficiency calibration (cb.effcal)
		is used.

		The functional form of the efficiency used is a modified version of the
		semi-empirical formula proposed by Vidmar (2001):
		eff(E) = c[0]*(1.0-exp(-mu(eng)*c[1]))*(tau(eng)+sigma(eng)*(1.0-exp(-(mu(eng)*c[3])**c[2]))*c[4])/mu(eng) if the effcal is length 5 or
		eff(E) = c[0]*exp(-mu_w(eng)*c[5])*exp(-mu(eng)*c[6])*(1.0-exp(-mu(eng)*c[1]))*(tau(eng)+sigma(eng)*(1.0-exp(-(mu(eng)*c[3])**c[2]))*c[4])/mu(eng)
		if the effcal is length 7.

		Parameters
		----------
		energy : array_like
			Peak energy in keV.

		effcal : array_like, optional
			Efficiency calibration parameters. length 5 or 7 array, depending on whether the efficiency
			fit includes the low-energy components.

		Returns
		-------
		efficiency : np.ndarray
			Absolute efficiency at the given energies.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.effcal)
		[1.87867638e-02 8.82671055e+01 2.09673703e+00 1.66193806e+00
		 3.90089610e-01 3.79361265e+00 1.24058591e-02]
		>>> print(cb.eff(50*np.arange(1,10)))
		[0.00469671 0.00553412 0.00502548 0.00436836 0.00369016 0.00315241
		 0.002754   0.00245546 0.00222482]

		"""

		energy = np.asarray(energy)

		if effcal is None:
			effcal = self.effcal


		if len(effcal)==5:
			sa, l, alpha, l0, kappa = tuple(effcal)
			return sa*(1.0-np.exp(-_MU(energy)*l))*(_TAU(energy)+_SIGMA(energy)*(1.0-np.exp(-(_MU(energy)*l0)**alpha))*kappa)/_MU(energy)

		else:
			sa, l, alpha, l0, kappa, w, d = tuple(effcal)
			return sa*np.exp(-_MU_W(energy)*w)*np.exp(-_MU(energy)*d)*(1.0-np.exp(-_MU(energy)*l))*(_TAU(energy)+_SIGMA(energy)*(1.0-np.exp(-(_MU(energy)*l0)**alpha))*kappa)/_MU(energy)

		
	def unc_eff(self, energy, effcal=None, unc_effcal=None):
		"""Uncertainty in the efficiency

		Returns the calculated uncertainty in efficiency for an input
		array of energies.  If `effcal` or `unc_effcal` are not none,
		they are used instead of the calibration object's internal
		values. (cb.unc_effcal)  unc_effcal must be a covariance matrix of the
		same dimension as effcal.

		Parameters
		----------
		energy : array_like
			Peak energy in keV.

		effcal : array_like, optional
			Efficiency calibration parameters. length 5 or 7 array, depending on whether the efficiency
			fit includes the low-energy components.

		unc_effcal : array_like, optional
			Efficiency calibration covariance matrix. shape 5x5 or 7x7, depending on the length of effcal.

		Returns
		-------
		unc_efficiency : np.ndarray
			Absolute uncertainty in efficiency for the given energies.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.eff(50*np.arange(1,10)))
		[0.00469671 0.00553412 0.00502548 0.00436836 0.00369016 0.00315241
		 0.002754   0.00245546 0.00222482]
		>>> print(cb.unc_eff(50*np.arange(1,10)))
		[1.84927072e-05 2.06720560e-05 4.00073148e-05 3.34061820e-05
		 1.49979136e-05 1.05818602e-05 9.72833934e-06 8.95311822e-06
		 8.10774373e-06]

		"""

		energy = np.asarray(energy)

		if effcal is None or unc_effcal is None:
			effcal, unc_effcal = self.effcal, self.unc_effcal

		eps = np.abs(1E-2*effcal)
		var = np.zeros(len(energy)) if energy.shape else 0.0

		for n in range(len(effcal)):
			for m in range(n, len(effcal)):

				if not np.isfinite(unc_effcal[n][m]):
					return np.inf*(var+1.0)

				c_n, c_m = np.copy(effcal), np.copy(effcal)
				c_n[n], c_m[m] = c_n[n]+eps[n], c_m[m]+eps[m]

				par_n = (self.eff(energy, c_n)-self.eff(energy, effcal))/eps[n]
				par_m = (self.eff(energy, c_m)-self.eff(energy, effcal))/eps[m]

				var += unc_effcal[n][m]*par_n*par_m*(2.0 if n!=m else 1.0)

		return np.sqrt(var)
		
	def res(self, channel, rescal=None):
		"""Resolution calibration

		Calculates the expected 1-sigma peak widths for a given input array
		of channel numbers.  If `rescal` is given, it is used instead of
		the calibration object's internal value (cb.rescal).

		Parameters
		----------
		channel : array_like
			Spectrum channel number.  The maximum channel number should
			be the length of the spectrum.

		rescal : array_like, optional
			Resolution calibration parameters.  length 2 array if resolution calibration is of the form
			R = a + b*chan (default), or length 1 if R = a*sqrt(chan).

		Returns
		-------
		resolution : np.ndarray
			Calculated 1-sigma width of the peaks given the input channel numbers.

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.rescal)
		[2.e+00 4.e-04]
		>>> print(cb.res(100*np.arange(1,10)))
		[2.04 2.08 2.12 2.16 2.2  2.24 2.28 2.32 2.36]

		"""

		channel = np.asarray(channel)

		if rescal is None:
			rescal = self.rescal

		if len(rescal)==1:
			return rescal[0]*np.sqrt(channel)

		elif len(rescal)==2:
			return rescal[0] + rescal[1]*channel
		
	def map_channel(self, energy, engcal=None):
		"""Energy to channel calibration

		Calculates the spectrum channel number corresponding to a given
		energy array.  This should return the inverse of the energy calibration, but
		as an integer-type channel number.

		Parameters
		----------
		energy : array_like
			Peak energy in keV

		engcal : array_like, optional
			Energy calibration parameters. length 2 or 3 array, depending on whether the calibration
			is linear or quadratic.

		Returns
		-------
		channel : np.ndarray
			Calculated channel number given the input energy array

		Examples
		--------
		>>> cb = ci.Calibration()
		>>> print(cb.engcal)
		[0.  0.3]
		>>> print(cb.map_channel(300))
		1000
		>>> print(cb.eng(cb.map_channel(300)))
		300.0

		"""

		energy = np.asarray(energy)

		if engcal is None:
			engcal = self.engcal

		if len(engcal)==3:
			if engcal[2]!=0.0:
				disc = engcal[1]**2-4.0*engcal[2]*(engcal[0]-energy)
				if np.any(disc<0.0):
					raise ValueError('Energy calibration {0} cannot be inverted at energy {1} (negative discriminant).'.format(list(engcal), np.atleast_1d(energy)[np.atleast_1d(disc<0.0)].tolist()))
				return np.array(np.rint(0.5*(np.sqrt(disc)-engcal[1])/engcal[2]), dtype=np.int32)

		if engcal[1]==0.0:
			raise ValueError('Energy calibration {} cannot be inverted (zero slope).'.format(list(engcal)))
		return np.array(np.rint((energy-engcal[0])/engcal[1]), dtype=np.int32)

	def _diag_row(self, name, chi2, dof, n_points, n_dropped, model, scale, fit, p0, unc, body, bounds=None, par_names=None):
		# One diagnostics row for a calibration sub-fit, flags per the shared
		# vocabulary. converged is True by construction: a sub-fit that fails
		# to converge raises out of calibrate().
		flags = []
		if bounds is not None:
			for n, side in _at_bound(fit, bounds[0], bounds[1]):
				flags.append('at_bound:'+par_names[n])
		if _unmoved(fit, p0):
			flags.append('unmoved')
		if not np.all(np.isfinite(np.asarray(unc, dtype=np.float64))):
			flags.append('singular_cov')
		if dof>0 and np.isfinite(chi2) and chi2>10.0:
			flags.append('chi2_high')
		return {'fit':name, 'chi2':chi2 if (dof>0 and np.isfinite(chi2)) else np.nan, 'dof':int(dof),
				'n_points':int(n_points), 'n_dropped':int(n_dropped), 'converged':True, 'model':model,
				'scale_factor':float(scale), 'flags':','.join(flags), 'message':body}

	@property
	def diagnostics(self):
		"""Fit diagnostics from the most recent calibration

		Read-only pd.DataFrame with one row per calibration sub-fit (`fit` is
		'engcal', 'rescal' or 'effcal') and columns chi2 (reduced, over all
		n_points that entered the fit - flagged outliers included), dof,
		n_points (points the fit used), n_dropped (pre-fit drops plus post-fit
		outlier clips; clipping affects the stored points, not the fit),
		converged, model, scale_factor (uncertainty inflation applied;
		1.0 = none), flags (comma-joined, e.g. 'chi2_high', 'at_bound:l') and
		message (the associated summary text).  Empty (with the full schema)
		before any calibration has been performed; rebuilt on each
		`calibrate()` call.  Accessing it never triggers a fit.
		"""
		if self._diagnostics is None:
			return _diagnostics_frame()
		return self._diagnostics.copy()

	def _tidy_points(self, group, x_col, y_col, unc_col, fn, str_cols=()):
		# Union of the used and dropped storage groups for one calibration as
		# a tidy table: point columns, provenance strings when stored, then
		# used / reason / residual (measured minus fitted, in the y-quantity).
		cols = [x_col, y_col, unc_col]+list(str_cols)
		empty = pd.DataFrame({**{c: pd.Series(dtype=(object if c in str_cols else float)) for c in cols},
							  'used': pd.Series(dtype=bool), 'reason': pd.Series(dtype=object),
							  'residual': pd.Series(dtype=float)})
		if group not in self._calib_data:
			return empty
		fit = self._calib_data[group].get('fit', None)
		frames = []
		for grp, is_used in [(group, True), (group+'_dropped', False)]:
			d = self._calib_data.get(grp, {})
			if x_col not in d or not len(np.atleast_1d(d[x_col])):
				continue
			n = len(np.atleast_1d(d[x_col]))
			f = {c:(np.atleast_1d(d[c]) if c in d else np.array(['']*n, dtype=object)) for c in cols}
			f['used'] = np.full(n, is_used, dtype=bool)
			f['reason'] = np.atleast_1d(d['reason']) if 'reason' in d else np.array(['']*n, dtype=object)
			y = np.asarray(f[y_col], dtype=np.float64)
			x = np.asarray(f[x_col], dtype=np.float64)
			f['residual'] = y-fn(x, fit) if fit is not None else np.full(n, np.nan)
			frames.append(pd.DataFrame(f))
		if not frames:
			return empty
		return pd.concat(frames, ignore_index=True)

	@property
	def engcal_data(self):
		"""Energy-calibration points from the most recent calibration

		Read-only pd.DataFrame with one row per measured point, including the
		points the fit rejected: columns channel, energy, unc_channel, used
		(False for rejected points), reason ('' when used, else e.g.
		'unc>25%') and residual (measured minus fitted energy, keV).  Empty
		(with the full schema) if no calibration data is present.
		"""
		return self._tidy_points('engcal', 'channel', 'energy', 'unc_channel', self.eng)

	@property
	def rescal_data(self):
		"""Resolution-calibration points from the most recent calibration

		Read-only pd.DataFrame with one row per measured point, including the
		points the fit rejected: columns channel, width, unc_width, used
		(False for rejected points - 'outlier chi2>10' rows entered the fit
		but are clipped from the stored calibration points),
		reason ('' when used, else 'unc>33%' or 'outlier chi2>10') and
		residual (measured minus fitted width, channels).  Empty (with the
		full schema) if no calibration data is present.
		"""
		return self._tidy_points('rescal', 'channel', 'width', 'unc_width', self.res)

	@property
	def effcal_data(self):
		"""Efficiency-calibration points from the most recent calibration

		Read-only pd.DataFrame with one row per measured point, including the
		points the fit rejected: columns energy, efficiency, unc_efficiency,
		isotope (source provenance - which isotope the point came from; ''
		for calibrations saved before it was recorded), used (False for
		rejected points - 'outlier chi2>10' rows entered the fit but are
		clipped from the stored calibration points), reason ('' when used,
		else 'unc>33%' or 'outlier chi2>10') and residual (measured minus
		fitted efficiency).  Empty (with the full schema) if no calibration
		data is present.
		"""
		return self._tidy_points('effcal', 'energy', 'efficiency', 'unc_efficiency', self.eff,
								 str_cols=('isotope',))

	def calibrate(self, spectra, sources):
		"""Generate calibration parameters from spectra

		Performs an energy, resolution and efficiency calibration on peak fits
		to a given list of spectra.  Reference activities must be given for the
		efficiency calibration.  Spectra are allowed to have isotopes that are
		not in `sources`, but these will not be included in the efficiency calibration.

		Parameters
		----------
		spectra : list of sp.Spectrum
			List of calibration spectra.  Must have sp.isotopes defined, and matching
			the isotopes given in `sources`.

		sources : str, list, dict or pd.DataFrame
			Datatype or (if str) file that can be converted into a
			pandas DataFrame.  Required keys are 'isotope', 'A0' (reference activity),
			and 'ref_date' (reference date).

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.7E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> print(cb.effcal)
		[5.02300989e-02 1.00000000e+02 2.82394962e+00 2.45721823e+00
		 2.91455256e-01]
		>>> cb.plot()

		"""

		if type(sources)==str:
			if sources.endswith('.json'):
				sources = pd.DataFrame(json.loads(open(sources).read()))
			elif sources.endswith('.csv'):
				sources = pd.read_csv(sources, header=0).ffill()
			elif sources.endswith('.db'):
				sources = pd.read_sql('SELECT * FROM sources', _get_connection(sources))
		else:
			if type(sources)==dict:
				if type(sources['A0'])==float:
					sources['A0'] = [sources['A0']]
			sources = pd.DataFrame(sources)
		try:
			sources['ref_date'] = pd.to_datetime(sources['ref_date'], format='%m/%d/%Y %H:%M:%S')
		except (ValueError, TypeError):
			for ip, rd in zip(sources['isotope'], sources['ref_date']):
				try:
					pd.to_datetime(rd, format='%m/%d/%Y %H:%M:%S')
				except (ValueError, TypeError):
					raise ValueError("Calibration.calibrate: could not parse ref_date {0!r} for source {1} - expected '%m/%d/%Y %H:%M:%S' (e.g. 01/01/2009 12:00:00)".format(rd, ip))
			raise

		src_itps = sources['isotope'].to_list()
		lm = {ip:Isotope(ip).dc for ip in src_itps}
		unc_lm = {ip:Isotope(ip).decay_const(unc=True)[1] for ip in src_itps}

		specs = []
		for n,sp in enumerate(spectra):
			if type(sp)==str:
				from .spectrum import Spectrum
				spec = Spectrum(sp)
				spec.isotopes = sources['isotope'].to_list()
				specs.append(spec)
			else:
				specs.append(sp)
		spectra = specs

		cb_dat, eff_meta = [], []
		for sp in spectra:
			if sp._fits is None:
				sp.fit_peaks()

			cfg, ix = sp.fit_config, -1

			for ft in sp._fits:
				f, u = ft['fit'], ft['unc']
				B = {'snip':0, 'constant':1, 'linear':2, 'quadratic':3}[cfg['bg'].lower()]
				L = 3+2*int(cfg['skew_fit'])+int(cfg['step_fit'])

				for n in range(int((len(f)-B)/L)):
					ix += 1
					pk = sp.peaks.loc[ix, :]
					if pk['isotope'] not in src_itps:
						continue

					j = B+n*L+1
					mu, sig = f[j], f[j+1]
					unc_mu, unc_sig = np.sqrt(u[j][j]), np.sqrt(u[j+1][j+1])
					eng = pk['energy']

					src = sources[sources['isotope']==pk['isotope']].reset_index(drop=True)
					rd, A0 = src.loc[0, 'ref_date'], src.loc[0, 'A0']
					unc_A0 = src.loc[0, 'unc_A0'] if 'unc_A0' in src.columns else 0.0
					td = (sp.start_time-rd).total_seconds()
					dc = lm[pk['isotope']]

					if sp._atten_corr is None:
						corr = sp._geom_corr
					else:
						corr = sp._geom_corr*sp._atten_corr(pk['energy'])

					eff = pk['counts']*dc/(corr*(1.0-np.exp(-dc*sp.real_time))*np.exp(-dc*td)*pk['intensity']*A0*(sp.live_time/sp.real_time))
					unc_eff = eff*np.sqrt((pk['unc_counts']/pk['counts'])**2+(pk['unc_intensity']/pk['intensity'])**2+(unc_lm[pk['isotope']]/dc)**2+(unc_A0/A0)**2)

					cb_dat.append([mu, eng, unc_mu, sig, unc_sig, eff, unc_eff])
					# per-point uncertainty decomposition for the efficiency fit's
					# covariance: counting is independent; the gamma intensity is
					# common within a line; the decay constant and reference
					# activity are common to every point of the source
					eff_meta.append({'rel_stat':pk['unc_counts']/pk['counts'],
									'rel_line':pk['unc_intensity']/pk['intensity'],
									'rel_src':np.sqrt((unc_lm[pk['isotope']]/dc)**2+(unc_A0/A0)**2),
									'line':pk['isotope']+':'+'{:.2f}'.format(eng),
									'src':pk['isotope']})

		if not len(cb_dat):
			raise ValueError('Calibration.calibrate: no calibration points - no fitted peaks matched the source isotopes [{0}]. Check sp.isotopes, the source isotope names, and that fit_peaks found peaks.'.format(', '.join(map(str, src_itps))))

		cb_dat = np.array(cb_dat)
		eff_meta = pd.DataFrame(eff_meta)
		self._calib_data = {'engcal':{'channel':cb_dat[:,0], 'energy':cb_dat[:,1], 'unc_channel':cb_dat[:,2]},
							'rescal':{'channel':cb_dat[:,0], 'width':cb_dat[:,3], 'unc_width':cb_dat[:,4]},
							'effcal':{'energy':cb_dat[:,1], 'efficiency':cb_dat[:,5], 'unc_efficiency':cb_dat[:,6]}}

		diag_rows = []

		x, y, yerr = self._calib_data['engcal']['channel'], self._calib_data['engcal']['energy'], self.eng(self._calib_data['engcal']['unc_channel'])
		n_tot = len(x)
		keep = (0.25*y>yerr)&(yerr>0.0)&(np.isfinite(yerr))
		self._calib_data['engcal_dropped'] = {'channel':x[~keep], 'energy':y[~keep], 'unc_channel':yerr[~keep],
											  'reason':np.array(['unc>25%']*int(np.sum(~keep)))}
		x, y, yerr = x[keep], y[keep], yerr[keep]
		if not len(x):
			raise ValueError('Calibration.calibrate: all {0} engcal points dropped (unc>25% of value) - cannot fit the energy calibration. Check the peak fits.'.format(n_tot))
		fn = lambda x, *A: self.eng(x, A)
		p0 = spectra[0].cb.engcal
		with np.errstate(all='ignore'):
			fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=p0)
		self._calib_data['engcal'] = {'channel':x, 'energy':y, 'unc_channel':yerr, 'fit':fit, 'unc':unc}
		self.engcal = fit
		dof = len(x)-len(fit)
		chi2 = float(np.sum((y-self.eng(x, fit))**2/yerr**2))/dof if dof>0 else np.inf
		model = 'quadratic' if len(fit)==3 else 'linear'
		body = 'engcal [{0}] fit to {1}/{2} points'.format(model, len(x), n_tot)
		if n_tot-len(x):
			body += ' ({0} dropped: unc>25% of value)'.format(n_tot-len(x))
		body += '; chi2/dof={0:.2g}'.format(chi2)
		_log.info('Calibration.calibrate: '+body)
		diag_rows.append(self._diag_row('engcal', chi2, dof, len(x), n_tot-len(x), model, 1.0, fit, p0, unc, body))

		x, y, yerr = self._calib_data['rescal']['channel'], self._calib_data['rescal']['width'], self._calib_data['rescal']['unc_width']
		n_tot = len(x)
		keep = (0.33*y>yerr)&(yerr>0.0)&(np.isfinite(yerr))
		drop = {'channel':x[~keep], 'width':y[~keep], 'unc_width':yerr[~keep]}
		x, y, yerr = x[keep], y[keep], yerr[keep]
		if not len(x):
			raise ValueError('Calibration.calibrate: all {0} rescal points dropped (unc>33% of value) - cannot fit the resolution calibration. Check the peak fits.'.format(n_tot))
		fn = lambda x, *A: self.res(x, A)
		p0 = spectra[0].cb.rescal
		with np.errstate(all='ignore'):
			fit, unc = curve_fit(fn, x, y, sigma=yerr, p0=p0)
		kept = (self.res(x, fit)-y)**2/yerr**2 < 10.0
		idx = np.where(kept)
		self._calib_data['rescal'] = {'channel':x[idx], 'width':y[idx], 'unc_width':yerr[idx], 'fit':fit, 'unc':unc}
		self._calib_data['rescal_dropped'] = {'channel':np.concatenate([drop['channel'], x[~kept]]),
											  'width':np.concatenate([drop['width'], y[~kept]]),
											  'unc_width':np.concatenate([drop['unc_width'], yerr[~kept]]),
											  'reason':np.array(['unc>33%']*len(drop['channel'])+['outlier chi2>10']*int(np.sum(~kept)))}
		self.rescal = fit
		n_out = len(x)-len(idx[0])
		# quoted over every point the fit used, flagged outliers included --
		# same convention as engcal and effcal (outlier clipping affects the
		# stored points, not the fit)
		dof = len(x)-len(fit)
		chi2 = float(np.sum((self.res(x, fit)-y)**2/yerr**2))/dof if dof>0 else np.inf
		model = 'linear' if len(fit)==2 else 'sqrt'
		body = 'rescal [{0}] fit to {1}/{2} points'.format(model, len(x), n_tot)
		if n_tot-len(x):
			body += ' ({0} dropped: unc>33% of value)'.format(n_tot-len(x))
		if n_out:
			body += '; {0} outliers: residual chi2>10'.format(n_out)
		body += '; chi2/dof={0:.2g}'.format(chi2)
		_log.info('Calibration.calibrate: '+body)
		diag_rows.append(self._diag_row('rescal', chi2, dof, len(x), (n_tot-len(x))+n_out, model, 1.0, fit, p0, unc, body))

		x, y, yerr = self._calib_data['effcal']['energy'], self._calib_data['effcal']['efficiency'], self._calib_data['effcal']['unc_efficiency']
		n_tot = len(x)
		keep = (0.33*y>yerr)&(yerr>0.0)&(np.isfinite(yerr))
		drop = {'energy':x[~keep], 'efficiency':y[~keep], 'unc_efficiency':yerr[~keep],
				'isotope':eff_meta['src'].to_numpy()[~keep], 'line':eff_meta['line'].to_numpy()[~keep]}
		x, y, yerr = x[keep], y[keep], yerr[keep]
		if not len(x):
			raise ValueError('Calibration.calibrate: all {0} effcal points dropped (unc>33% of value) - cannot fit the efficiency calibration. Check the peak fits and source activities.'.format(n_tot))
		meta = eff_meta[keep].reset_index(drop=True)
		fn = lambda x, *A: self.eff(x, A)

		def eff_cov(m):
			# covariance of the efficiency points, with magnitudes from the fitted
			# model values (measurement-weighted correlated fits bias low)
			V = np.diag((m*meta['rel_stat'].to_numpy())**2)
			for key, rel in [('line','rel_line'), ('src','rel_src')]:
				r = m*meta[rel].to_numpy()
				for g in meta[key].unique():
					u = np.where((meta[key]==g).to_numpy(), r, 0.0)
					V += np.outer(u, u)
			d = np.diag(V)
			V[np.diag_indices_from(V)] = d + 1E-12*np.max(d)
			return V

		def fit_eff(p0_, bounds_):
			# numeric warnings from the optimizer carry no context; failures
			# re-emerge as curie messages at the call sites
			with np.errstate(all='ignore'):
				f0, _ = curve_fit(fn, x, y, sigma=yerr, p0=p0_, bounds=bounds_, absolute_sigma=True)
				m0 = self.eff(x, f0)
				Vi = np.diag((m0*meta['rel_stat'].to_numpy())**2)
				V = eff_cov(m0)
				# one-sided scale factor on the INDEPENDENT component only: inconsistency
				# between points cannot indict the correlated modes
				S2, f1, u1, chi2n, chi2_0 = 1.0, f0, None, np.inf, None
				for _ in range(4):
					Vp = V if S2==1.0 else V+(S2-1.0)*Vi
					f1, u1 = curve_fit(fn, x, y, sigma=Vp, p0=f1, bounds=bounds_, absolute_sigma=True)
					r = y-self.eff(x, f1)
					chi2n = float(r @ np.linalg.solve(Vp, r))/max(len(y)-len(f1), 1)
					if chi2_0 is None:
						# goodness of fit before any inflation: this is what gets
						# reported (the converged chi2n tends to 1 by construction)
						chi2_0 = chi2n
					if not (np.isfinite(chi2n) and chi2n>1.0):
						break
					S2 = S2*chi2n
			return f1, u1, chi2n, chi2_0, S2

		p0 = spectra[0].cb.effcal
		p0 = p0.tolist() if len(p0)==7 else p0.tolist()+[0.5, 0.001]
		p0[0] = max([min([p0[0]*np.average(y/self.eff(x, p0), weights=(self.eff(x, p0)/yerr)**2),4.99]),0.0001])
		bounds = ([0.0, 0.0, 0.1, 0.0, 0.0, 0.001, 1E-12], [12, 100, 6, 50, 6, 6, 0.2])

		p0_used, bounds_used = p0[:5], (bounds[0][:5], bounds[1][:5])
		if any([sp.fit_config['xrays'] for sp in spectra]):
			try:
				fit5, unc5, chi5, chi5_0, S2_5 = fit_eff(p0[:5], (bounds[0][:5], bounds[1][:5]))
				fit7, unc7, chi7, chi7_0, S2_7 = fit_eff(fit5.tolist()+p0[5:], bounds)
				## Invert to find which is closer to one: selection uses the converged
				## whitened statistic; the messages report the pre-inflation chi2/dof
				c7 = chi7 if chi7>1.0 else 1.0/chi7
				c5 = chi5 if chi5>1.0 else 1.0/chi5
				fit, unc, eff_chi2, eff_S2 = (fit5, unc5, chi5_0, S2_5) if c5<=c7 else (fit7, unc7, chi7_0, S2_7)
				if c5>c7:
					p0_used, bounds_used = fit5.tolist()+p0[5:], bounds
				_log.info('Calibration.calibrate: effcal model selection: vidmar-{0} (chi2/dof={1:.2g}) preferred over vidmar-{2} (chi2/dof={3:.2g})'.format(
					*((5, chi5_0, 7, chi7_0) if c5<=c7 else (7, chi7_0, 5, chi5_0))))
			except Exception as err:
				_log.warning('Calibration.calibrate: effcal 7-parameter fit failed ({0}); using the 5-parameter form'.format(err))
				fit, unc, _, eff_chi2, eff_S2 = fit_eff(p0[:5], (bounds[0][:5], bounds[1][:5]))

		else:
			fit, unc, _, eff_chi2, eff_S2 = fit_eff(p0[:5], (bounds[0][:5], bounds[1][:5]))

		with np.errstate(all='ignore'):
			kept = (self.eff(x, fit)-y)**2/yerr**2 < 10.0
		idx = np.where(kept)
		self._calib_data['effcal'] = {'energy':x[idx], 'efficiency':y[idx], 'unc_efficiency':yerr[idx],
									  'isotope':meta['src'].to_numpy()[idx], 'line':meta['line'].to_numpy()[idx],
									  'fit':fit, 'unc':unc}
		self._calib_data['effcal_dropped'] = {'energy':np.concatenate([drop['energy'], x[~kept]]),
											  'efficiency':np.concatenate([drop['efficiency'], y[~kept]]),
											  'unc_efficiency':np.concatenate([drop['unc_efficiency'], yerr[~kept]]),
											  'isotope':np.concatenate([drop['isotope'], meta['src'].to_numpy()[~kept]]),
											  'line':np.concatenate([drop['line'], meta['line'].to_numpy()[~kept]]),
											  'reason':np.array(['unc>33%']*len(drop['energy'])+['outlier chi2>10']*int(np.sum(~kept)))}
		self.effcal = fit
		self.unc_effcal = unc

		outliers = ['{0} {1:.1f}'.format(ln.split(':')[0], float(ln.split(':')[1])) for ln in meta[~kept]['line']]
		model = 'vidmar-{0}'.format(len(fit))
		body = 'effcal [{0}] fit to {1}/{2} points'.format(model, len(x), n_tot)
		if n_tot-len(x):
			body += ' ({0} dropped: unc>33% of value)'.format(n_tot-len(x))
		if len(outliers):
			body += '; {0} outliers: residual chi2>10 [{1} keV]'.format(len(outliers), ', '.join(outliers))
		body += '; chi2/dof={0:.2g}'.format(eff_chi2)
		eff_scale = float(np.sqrt(eff_S2)) if np.isfinite(eff_S2) and eff_S2>1.0 else 1.0
		if eff_scale>1.0:
			body += '; independent uncertainty components scaled x{0:.2f}'.format(eff_scale)
		_log.info('Calibration.calibrate: '+body)
		diag_rows.append(self._diag_row('effcal', eff_chi2, len(x)-len(fit), len(x), (n_tot-len(x))+len(outliers),
										model, eff_scale, fit, p0_used, unc, body, bounds=bounds_used,
										par_names=['sa','l','alpha','l0','kappa','w','d']))
		self._diagnostics = _diagnostics_frame(diag_rows)

		for sp in spectra:
			sp.cb.engcal = self._calib_data['engcal']['fit']
			sp.cb.rescal = self._calib_data['rescal']['fit']
			sp.cb.effcal = self._calib_data['effcal']['fit']
			sp.cb.unc_effcal = self._calib_data['effcal']['unc']
			sp._peaks, sp._fits = None, None



		
	def saveas(self, filename):
		"""Save the calibration as a .json file

		Saves the energy, resolution and efficiency calibration to a .json
		file, which can be recalled by a new calibration object by passing
		the 'filename' keyword to Calibration() upon construction.

		Parameters
		----------
		filename : str
			Complete filename to save the calibration. Must end in '.json'.

		Examples
		--------
		>>> sp = ci.Spectrum('eu_calib_7cm.Spe')
		>>> sp.isotopes = ['152EU']

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.7E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> print(cb.effcal)
		[5.02300989e-02 1.00000000e+02 2.82394962e+00 2.45721823e+00
		 2.91455256e-01]
		>>> cb.saveas('example_calib.json')

		>>> cb = ci.Calibration('example_calib.json')
		>>> print(cb.effcal)
		[5.02300989e-02 1.00000000e+02 2.82394962e+00 2.45721823e+00
		 2.91455256e-01]

		"""

		if any([filename.endswith(e) for e in ['.png','.pdf','.eps','.pgf','.ps','.raw','.rgba','.svg','.svgz']]):
				self.plot(saveas=filename, show=False)

		if filename.endswith('.json'):
			js = {'engcal':self.engcal.tolist(),
				  'effcal':self.effcal.tolist(),
				  'unc_effcal':self.unc_effcal.tolist(),
				  'rescal':self.rescal.tolist()}
			if self._calib_data:
				js['_calib_data'] = {}
				for cl in _CALIB_GROUPS:
					if cl in self._calib_data:
						js['_calib_data'][cl] = {i:np.asarray(self._calib_data[cl][i]).tolist() for i in self._calib_data[cl]}

			with open(filename, 'w') as f:
				json.dump(js, f, indent=4)

		
	def plot_engcal(self, **kwargs):
		"""Plot the energy calibration

		Draws the energy calibration, with measurements from peak fit data if available.

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

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_engcal()

		"""

		f, ax = _init_plot(**kwargs)

		if 'engcal' in self._calib_data:
			d = self._calib_data['engcal']
			ax.errorbar(d['energy'], d['channel'], yerr=d['unc_channel'], ls='None', marker='o')
			x = np.arange(min(d['energy']), max(d['energy']), 0.1)
			ax.plot(x, self.map_channel(x, d['fit']))

		else:
			x = np.arange(20, 3000, 0.1)
			ax.plot(x, self.map_channel(x))

		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('ADC Channel')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_rescal(self, **kwargs):
		"""Plot the resolution calibration

		Draws the resolution calibration, with measurements from peak fit data if available.

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

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_rescal()

		"""

		f, ax = _init_plot(**kwargs)

		if 'rescal' in self._calib_data:
			d = self._calib_data['rescal']
			ax.errorbar(d['channel'], d['width'], yerr=d['unc_width'], ls='None', marker='o')
			x = np.arange(min(d['channel']), max(d['channel']), 0.1)
			ax.plot(x, self.res(x, d['fit']))

		else:
			x = np.arange(0, 2**14, 0.1)
			ax.plot(x, self.res(x))

		ax.set_xlabel('ADC Channel')
		ax.set_ylabel('Peak Width')

		return _draw_plot(f, ax, **kwargs)
		
	def plot_effcal(self, **kwargs):
		"""Plot the efficiency calibration

		Draws the efficiency calibration, with measurements from peak fit data if available.

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

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot_effcal()

		"""
		cm = colormap()
		cm_light = colormap(shade='light')

		f, ax = _init_plot(**kwargs)

		if 'effcal' in self._calib_data:
			d = self._calib_data['effcal']
			ax.errorbar(d['energy'], d['efficiency'], yerr=d['unc_efficiency'], ls='None', marker='o', color=cm['k'])
			x = np.arange(min(d['energy']), max(d['energy']), 0.1)
			ax.plot(x, self.eff(x, d['fit']), color=cm['k'])
			low = self.eff(x, d['fit'])-self.unc_eff(x, d['fit'], d['unc'])
			high = self.eff(x, d['fit'])+self.unc_eff(x, d['fit'], d['unc'])
			ax.fill_between(x, low, high, facecolor=cm_light['k'], alpha=0.5)

		else:
			x = np.arange(20, 3000, 0.1)
			ax.plot(x, self.eff(x), color=cm['k'])
			low = self.eff(x)-self.unc_eff(x)
			high = self.eff(x)+self.unc_eff(x)
			ax.fill_between(x, low, high, facecolor=cm_light['k'], alpha=0.5)

		ax.set_xlabel('Energy (keV)')
		ax.set_ylabel('Efficiency (abs.)')

		return _draw_plot(f, ax, **kwargs)
		
	def plot(self, **kwargs):
		"""Plots energy, resolution and efficiency calibrations

		Draws all three of energy, resolution and efficiency calibrations on a 
		single figure, and shows measured values from peak data, if available.

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

		>>> cb = ci.Calibration()
		>>> cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
		>>> cb.plot()

		"""

		
		f, ax = _init_plot(_N_plots=3, figsize=(12.8, 4.8), **kwargs)

		f, ax[0] = self.plot_engcal(f=f, ax=ax[0], show=False, return_plot=True)
		ax[0].set_title('Energy Calibration')
		f, ax[1] = self.plot_rescal(f=f, ax=ax[1], show=False, return_plot=True)
		ax[1].set_title('Resolution Calibration')
		f, ax[2] = self.plot_effcal(f=f, ax=ax[2], show=False, return_plot=True)
		ax[2].set_title('Efficiency Calibration')

		return _draw_plot(f, ax, **kwargs)
		

