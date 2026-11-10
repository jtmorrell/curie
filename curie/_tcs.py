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
  is at or above the floor E_min, of 1 - (pair / line) * eps_t(partner),
  except that the x-ray partners of a gamma-ray or annihilation line enter as
  one factor, 1 - sum((pair / line) * eps_t(partner));
- gain is the sum, over pairs of coincident gamma rays (E1, E2) with both
  energies at or above E_min that add up to the line energy within sum_tol,
  of (pair / line) * eps_p(E1) * eps_p(E2) / eps_p(line) * loss_12 *
  (1 - q * eps_t(line)).  Here loss_12 is the loss of the line without the
  factors of its partners at E1 and E2, and q = min(pair(line, E1),
  pair(line, E2), pair) / pair is the fraction of the pair's decays that
  also emit the line photon.  A pair whose summed energy lies within sum_tol
  of several lines adds only to the nearest one, so its summed-in counts are
  counted once.

A summed-in count leaves the peak when any other photon of the same decay
deposits energy.  Those photons are taken to be the partners of the line
other than the two summing photons, which cannot remove their own sum, and
the line photon itself where the decay emits it with the pair.  The pairwise
table holds no three-photon intensities, so q is their upper bound; it is
exact for a single cascade and zero for a crossover, whose photon is never
emitted with the pair.  For a crossover inside a cascade the photons in
coincidence with the pair are those in coincidence with the line, and the
factor is loss * (1 + gain).

The x-ray partners share one factor because a vacancy is filled by at most one
K x-ray, so the x-rays of one vacancy are exclusive and their summing-out
probabilities add.  On a gamma-ray or annihilation line this is exact for one
vacancy per decay and leaves an error of second order where electron capture
and internal conversion both leave one.  On an x-ray line the partners of type
'XR' are gamma rays, which stay one factor each; the table pairs no two
x-rays, so where a decay leaves two vacancies the loss of an x-ray line misses
the x-ray of the other vacancy, an error of first order in the efficiencies.

The product form is exact for a single cascade; on a branched scheme the
pairwise table carries no correlation between three or more photons, which
leaves an error of second order in the efficiencies.  Angular correlations are
not included.

Two table conventions enter the factor.  A pair of equal-energy gamma rays in
cascade is stored once, so it counts twice toward the loss of that line.  A
pair with an annihilation photon (type '511') counts both quanta of each
annihilation; the two quanta leave back to back, so at most one reaches a
detector that does not surround the source, which makes their loss additive, a
single factor in the product, and the (511, 511) self-pair contributes neither
loss nor gain.
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
		Columns 'energy' and 'coinc_energy' (keV), 'coinc_type' (the class of
		the pair, the same under both orderings: 'g' when both photons are gamma
		rays, 'XR' when it holds an x-ray, '511' when it holds an annihilation
		photon), and 'intensity' and 'unc_intensity'
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


def _factors(lines, pairs, eps_p, eps_t, E_min=15.0, sum_tol=1.0, *, xray_lines):
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
		and a line energy for the pair to add to that line; a pair adds only to
		the line nearest its summed energy.  Default 1.0.

	xray_lines : array_like
		Energies of the x-ray lines among `lines` (empty when there are none),
		whose 'XR' partners are gamma rays and enter the loss one factor each.
		Required, since an x-ray line taken for a gamma-ray line would have its
		gamma-ray partners combined into one factor.

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
	s = np.zeros(len(E))
	p = pairs[(pairs['coinc_energy']>=E_min)&pairs['energy'].isin(list(pos))]
	p = p[~((p['coinc_type']=='511')&(p['energy']==p['coinc_energy']))]
	idx = np.array([pos[e] for e in p['energy']], dtype=int)
	ce, P = p['coinc_energy'].to_numpy(dtype=float), p['intensity'].to_numpy(dtype=float)
	fac, single = np.ones(len(p)), np.zeros(len(p), dtype=bool)
	if len(p):
		ratio = P/I[idx]
		ratio[((p['coinc_type']=='g')&(p['energy']==p['coinc_energy'])).to_numpy()] *= 2.0
		x = ratio*np.asarray(eps_t(ce), dtype=float)
		xr = (p['coinc_type']=='XR').to_numpy()&~p['energy'].isin(list(xray_lines)).to_numpy()
		fac, single = 1.0-x, ~xr
		np.multiply.at(loss, idx[single], fac[single])
		np.add.at(s, idx[xr], x[xr])
		loss *= 1.0-s
	# single-partner rows that can hold a photon of a summing gamma-ray pair
	match = single&(p['coinc_type']!='511').to_numpy()

	gain = np.zeros(len(E))
	g = pairs[(pairs['coinc_type']=='g')&(pairs['energy']<=pairs['coinc_energy'])&(pairs['energy']>=E_min)]
	if len(g) and len(E):
		e1, e2 = g['energy'].to_numpy(dtype=float), g['coinc_energy'].to_numpy(dtype=float)
		k = np.abs(e1+e2-E[:,np.newaxis]).argmin(axis=0)
		hit = np.abs(e1+e2-E[k])<=sum_tol
		if hit.any():
			e1, e2, k, Pg = e1[hit], e2[hit], k[hit], g['intensity'].to_numpy(dtype=float)[hit]
			w = Pg*np.asarray(eps_p(e1), dtype=float)*np.asarray(eps_p(e2), dtype=float)
			et_line = np.asarray(eps_t(E[k]), dtype=float)
			for j in range(len(k)):
				on = idx==k[j]
				own = on&match&((ce==e1[j])|(ce==e2[j]))
				q = min(P[own&(ce==e1[j])].sum(), P[own&(ce==e2[j])].sum(), Pg[j])/Pg[j]
				w[j] *= (1.0-s[k[j]])*np.prod(fac[on&single&~own])*(1.0-q*et_line[j])
			np.add.at(gain, k, w)
			rows = np.unique(k)
			gain[rows] /= I[rows]*np.asarray(eps_p(E[rows]), dtype=float)

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
	xray_lines = set(lines['energy'])-set(isotope.gammas()['energy'])
	F = dict(zip(lines['energy'], _factors(lines, _pairs(isotope), eps_p, eps_t, E_min=E_min, sum_tol=sum_tol, xray_lines=xray_lines)))

	energy = np.asarray(energy, dtype=float)
	for e in energy.ravel():
		if e not in F:
			raise ValueError('{} keV is not a line energy of {} in the decay data.'.format(e, isotope.name))
	out = np.array([F[e] for e in energy.ravel()]).reshape(energy.shape)
	return float(out) if out.ndim==0 else out
