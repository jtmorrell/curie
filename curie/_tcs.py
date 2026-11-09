"""True-coincidence-summing correction factors from the pairwise coincidence table.

A gamma ray emitted in cascade with other photons loses full-energy-peak counts
when a coincident photon deposits energy in the detector at the same time
(summing out), and its peak gains counts when two coincident gamma rays whose
energies add up to the line energy are both fully absorbed (summing in).  The
decay database's `coincidences` table lists, for each decaying state, the
coincident photon pairs and the number of such pairs per 100 decays.

The factor F of a line is the ratio of its full-energy-peak count rate with
coincidence summing to the rate without it, so an observed peak area is the
summing-free area times F.  With line and pair intensities in percent,
F = loss + gain, where

- loss is the product, over the pairs keyed on the line whose partner energy
  is at or above the floor E_min, of 1 - (pair / line) * eps_t(partner);
- gain is the sum, over pairs of coincident gamma rays with both energies at
  or above E_min that add up to the line energy within sum_tol, of
  (pair / line) * eps_p(E1) * eps_p(E2) / eps_p(line).

The product form is exact for a single cascade; on a branched scheme the
pairwise table carries no correlation between three or more photons, which
leaves an error of second order in the efficiencies.  Angular correlations are
not included.

Two table conventions enter the factor.  A pair of equal-energy gamma rays in
cascade is stored once, so it counts twice toward the loss of that line.  An
annihilation partner (type '511') counts both quanta of each annihilation; the
two quanta leave back to back, so at most one reaches a detector that does not
surround the source, which makes their loss additive, a single factor in the
product, and the (511, 511) self-pair contributes neither loss nor gain.
"""
import numpy as np
import pandas as pd

from .data import _get_connection
from .isotope import Isotope


def _pairs(isotope):
	"""Coincident photon pairs of a decaying state, from the decay database.

	Parameters
	----------
	isotope : Isotope or str
		The decaying state, as an Isotope or any name Isotope accepts.

	Returns
	-------
	pd.DataFrame
		Columns 'energy' and 'coinc_energy' (keV), 'coinc_type' ('g', 'XR' or
		'511', the type of the partner), and 'intensity' and 'unc_intensity'
		(pairs per 100 decays; NaN where no uncertainty is given).  Each pair is
		listed under both orderings of its energies, except a pair of
		equal-energy photons, which is listed once.  A state without coincident
		photons gives an empty frame.

	"""

	if not isinstance(isotope, Isotope):
		isotope = Isotope(isotope)
	df = pd.read_sql('SELECT energy, coinc_energy, coinc_type, intensity, unc_intensity FROM coincidences WHERE isotope=? AND isomer=? ORDER BY energy, coinc_energy, coinc_type',
					_get_connection('decay'), params=(str(isotope.A)+isotope.element.upper(), isotope.isomer))
	return df.astype({'energy':float, 'coinc_energy':float, 'intensity':float, 'unc_intensity':float})


def _factors(lines, pairs, eps_p, eps_t, E_min=15.0, sum_tol=1.0):
	"""Coincidence-summing factor of every line of a decaying state.

	Parameters
	----------
	lines : pd.DataFrame
		Every line of the state, with columns 'energy' (keV) and 'intensity'
		(percent).  Rows at the same energy are one line, whose intensity is
		their sum.

	pairs : pd.DataFrame
		The state's coincident pairs, as `_pairs` returns them.

	eps_p, eps_t : callable
		Absolute full-energy-peak and total efficiencies, each taking an array
		of energies in keV.

	E_min : float, optional
		Partner energy floor, in keV: photons below it neither remove counts from
		a peak nor add to one.  Default 15.0.

	sum_tol : float, optional
		Largest difference, in keV, between the summed energy of a gamma-ray pair
		and a line energy for the pair to add to that line.  Default 1.0.

	Returns
	-------
	np.ndarray
		One factor per row of `lines`, in order; 1.0 exactly for a line with no
		coincident partner above the floor and no pair summing to it.

	"""

	I_line = lines.groupby('energy')['intensity'].sum()
	E, I = I_line.index.to_numpy(dtype=float), I_line.to_numpy(dtype=float)
	pos = {e:n for n,e in enumerate(E)}

	loss = np.ones(len(E))
	p = pairs[(pairs['coinc_energy']>=E_min)&pairs['energy'].isin(list(pos))]
	p = p[~((p['coinc_type']=='511')&(p['energy']==p['coinc_energy']))]
	if len(p):
		idx = np.array([pos[e] for e in p['energy']], dtype=int)
		ratio = p['intensity'].to_numpy(dtype=float)/I[idx]
		ratio[((p['coinc_type']=='g')&(p['energy']==p['coinc_energy'])).to_numpy()] *= 2.0
		np.multiply.at(loss, idx, 1.0-ratio*np.asarray(eps_t(p['coinc_energy'].to_numpy(dtype=float)), dtype=float))

	gain = np.zeros(len(E))
	g = pairs[(pairs['coinc_type']=='g')&(pairs['energy']<=pairs['coinc_energy'])&(pairs['energy']>=E_min)]
	if len(g):
		e1, e2 = g['energy'].to_numpy(dtype=float), g['coinc_energy'].to_numpy(dtype=float)
		near = np.abs(e1+e2-E[:,np.newaxis])<=sum_tol
		rows, cols = near.any(axis=1), near.any(axis=0)
		if rows.any():
			w = g['intensity'].to_numpy(dtype=float)[cols]*np.asarray(eps_p(e1[cols]), dtype=float)*np.asarray(eps_p(e2[cols]), dtype=float)
			gain[rows] = near[rows][:,cols].astype(float).dot(w)/(I[rows]*np.asarray(eps_p(E[rows]), dtype=float))

	F = loss+gain
	return F[np.array([pos[e] for e in lines['energy']], dtype=int)]


def _summing_factor(isotope, energy, eps_p, eps_t, E_min=15.0, sum_tol=1.0):
	"""Coincidence-summing factor of lines of a decaying state.

	Parameters
	----------
	isotope : Isotope or str
		The decaying state, as an Isotope or any name Isotope accepts.

	energy : float or array_like
		Line energies, in keV, each exactly equal to an energy in the state's
		gamma-ray table (x-ray and annihilation lines included).

	eps_p, eps_t : callable
		Absolute full-energy-peak and total efficiencies, each taking an array
		of energies in keV.

	E_min, sum_tol : float, optional
		Partner energy floor and summing tolerance, in keV, as in `_factors`.

	Returns
	-------
	float or np.ndarray
		The factor of each requested line: a float for a scalar `energy`, an
		array of the same shape otherwise.

	Raises
	------
	ValueError
		If an energy is not a line energy of the state.

	"""

	if not isinstance(isotope, Isotope):
		isotope = Isotope(isotope)
	lines = isotope.gammas(xrays=True)
	F = dict(zip(lines['energy'], _factors(lines, _pairs(isotope), eps_p, eps_t, E_min=E_min, sum_tol=sum_tol)))

	energy = np.asarray(energy, dtype=float)
	for e in energy.ravel():
		if e not in F:
			raise ValueError('{} keV is not a line energy of {} in the decay data.'.format(e, isotope.name))
	out = np.array([F[e] for e in energy.ravel()]).reshape(energy.shape)
	return float(out) if out.ndim==0 else out
