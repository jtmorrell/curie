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

A total efficiency above 1 is not physical; one used in the factor, for a
partner or for the line photon, is taken as at most 1.

A pair cannot occur more often than either of its photons, so in a consistent
table the ratio of a pair intensity to the intensity of its line is at most 1
(a 511 keV partner of a line other than 511 keV counts both quanta of each
annihilation, so its ratio is halved; a pair of two photons of the line energy
gives two photons of the line, so its ratio is doubled).  A line whose largest ratio, over its
pairs with a partner at or above E_min, exceeds 1.01 has table entries that
disagree, and its factor is NaN: the consumer leaves that line uncorrected.
"""
import numpy as np
import pandas as pd

from .data import _get_connection
from .isotope import Isotope

# largest pair-to-line ratio of a line whose factor is computed; the excess
# over 1 leaves room for the rounding of the tabulated intensities
_GUARD_RATIO = 1.01


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


def _largest_ratio(E, I, pairs, E_min):
	"""Largest pair-to-line ratio of each line energy E (intensities I), over the
	pairs keyed on the line with a partner at or above E_min; 0 for a line with
	none.  The (511, 511) self-pair is left out, a 511 keV partner of a line
	other than 511 keV is halved, and a gamma-ray pair of two photons of the
	line energy is doubled, as the factor counts it twice."""

	pos = {e:n for n,e in enumerate(E)}
	p = pairs[(pairs['coinc_energy']>=E_min)&pairs['energy'].isin(list(pos))]
	p = p[~((p['coinc_type']=='511')&(p['energy']==p['coinc_energy']))]
	ratio = np.zeros(len(E))
	if len(p):
		idx = np.array([pos[e] for e in p['energy']], dtype=int)
		half = ((p['coinc_type']=='511')&(p['coinc_energy']==511.0)&(p['energy']!=511.0)).to_numpy()
		twice = ((p['coinc_type']=='g')&(p['energy']==p['coinc_energy'])).to_numpy()
		np.maximum.at(ratio, idx, p['intensity'].to_numpy(dtype=float)*np.where(twice, 2.0, 1.0)/np.where(half, 2.0, 1.0)/I[idx])
	return ratio


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
		coincident partner above the floor and no pair summing to it, and NaN
		for a line whose largest pair-to-line ratio exceeds 1.01.

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
		x = ratio*np.minimum(np.asarray(eps_t(ce), dtype=float), 1.0)
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
			et_line = np.minimum(np.asarray(eps_t(E[k]), dtype=float), 1.0)
			for j in range(len(k)):
				on = idx==k[j]
				own = on&match&((ce==e1[j])|(ce==e2[j]))
				q = min(P[own&(ce==e1[j])].sum(), P[own&(ce==e2[j])].sum(), Pg[j])/Pg[j]
				w[j] *= (1.0-s[k[j]])*np.prod(fac[on&single&~own])*(1.0-q*et_line[j])
			np.add.at(gain, k, w)
			rows = np.unique(k)
			gain[rows] /= I[rows]*np.asarray(eps_p(E[rows]), dtype=float)

	F = loss+gain
	F[_largest_ratio(E, I, pairs, E_min)>_GUARD_RATIO] = np.nan
	return F[np.array([pos[e] for e in lines['energy']], dtype=int)]


class _Arrays(object):
	"""The factor of every line of a decaying state, from arrays built once.

	The arrays hold what `_factors` reads from the line and pair tables, so
	that an evaluation does arithmetic only.  `terms` gives the factor of each
	line as `_factors` does, with the efficiency of the x-ray partners of a
	gamma-ray or annihilation line multiplied by a scale, and the size of the
	correction.

	Parameters
	----------
	lines, pairs, xray_lines, E_min, sum_tol
		As in `_factors`.

	Attributes
	----------
	energy : np.ndarray
		The line energies of the state (keV), sorted and unique: the order of
		the values `terms` returns.

	grid : np.ndarray
		The energies (keV), sorted and unique, at which `terms` takes the
		efficiencies: the line energies and every partner energy.

	"""

	def __init__(self, lines, pairs, xray_lines, E_min=15.0, sum_tol=1.0):
		I_line = lines.groupby('energy')['intensity'].sum()
		E, I = I_line.index.to_numpy(dtype=float), I_line.to_numpy(dtype=float)
		pos = {e:n for n,e in enumerate(E)}
		self.energy, self._I, self._pos = E, I, pos

		# loss rows, as in _factors
		p = pairs[(pairs['coinc_energy']>=E_min)&pairs['energy'].isin(list(pos))]
		p = p[~((p['coinc_type']=='511')&(p['energy']==p['coinc_energy']))]
		idx = np.array([pos[e] for e in p['energy']], dtype=int)
		ce, P = p['coinc_energy'].to_numpy(dtype=float), p['intensity'].to_numpy(dtype=float)
		ratio = P/I[idx]
		ratio[((p['coinc_type']=='g')&(p['energy']==p['coinc_energy'])).to_numpy(dtype=bool)] *= 2.0
		xr = (p['coinc_type']=='XR').to_numpy(dtype=bool)&~p['energy'].isin(list(xray_lines)).to_numpy(dtype=bool)
		single = ~xr
		match = single&(p['coinc_type']!='511').to_numpy(dtype=bool)
		self._idx, self._ratio, self._xr, self._single = idx, ratio, xr, single

		# gain rows: each summing pair with its line, q, and the loss rows of
		# the line other than its own summing photons
		g = pairs[(pairs['coinc_type']=='g')&(pairs['energy']<=pairs['coinc_energy'])&(pairs['energy']>=E_min)]
		k, e1, e2, Pg = np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), np.zeros(0)
		if len(g) and len(E):
			e1, e2 = g['energy'].to_numpy(dtype=float), g['coinc_energy'].to_numpy(dtype=float)
			k = np.abs(e1+e2-E[:,np.newaxis]).argmin(axis=0)
			hit = np.abs(e1+e2-E[k])<=sum_tol
			e1, e2, k, Pg = e1[hit], e2[hit], k[hit], g['intensity'].to_numpy(dtype=float)[hit]
		q = np.zeros(len(k))
		row, other = [], []
		for j in range(len(k)):
			on = idx==k[j]
			own = on&match&((ce==e1[j])|(ce==e2[j]))
			q[j] = min(P[own&(ce==e1[j])].sum(), P[own&(ce==e2[j])].sum(), Pg[j])/Pg[j]
			m = np.flatnonzero(on&single&~own)
			row.extend([j]*len(m))
			other.extend(m)
		self._k, self._Pg, self._q = k, Pg, q
		self._row, self._other = np.array(row, dtype=int), np.array(other, dtype=int)
		self._rows = np.unique(k)

		self.grid = np.unique(np.concatenate([E, ce, e1, e2]))
		self._i_line = np.searchsorted(self.grid, E)
		self._i_ce = np.searchsorted(self.grid, ce)
		self._i_e1, self._i_e2 = np.searchsorted(self.grid, e1), np.searchsorted(self.grid, e2)
		self._guarded = _largest_ratio(E, I, pairs, E_min)>_GUARD_RATIO

	def index(self, energy):
		"""Position of each energy in `energy` (keV), -1 where it is not a line
		energy of the state."""

		return np.array([self._pos.get(e, -1) for e in np.atleast_1d(np.asarray(energy, dtype=float))], dtype=int)

	def terms(self, ep, et, scale=1.0):
		"""Factor and size of the correction of every line.

		Parameters
		----------
		ep, et : np.ndarray
			Absolute full-energy-peak and total efficiencies at `grid`.

		scale : float, optional
			Multiplies the total efficiency of each x-ray partner of a
			gamma-ray or annihilation line; 0 leaves those partners out.
			Default 1.

		Returns
		-------
		F, c : np.ndarray
			Per line of `energy`: the factor F = loss + gain, and the size of the
			correction c = (1 - loss) + gain; both NaN for a line whose largest
			pair-to-line ratio exceeds 1.01.

		"""

		ep, et = np.asarray(ep, dtype=float), np.asarray(et, dtype=float)
		n = len(self.energy)
		loss, s = np.ones(n), np.zeros(n)
		etc = et[self._i_ce]
		etc = np.minimum(np.where(self._xr, etc*scale, etc), 1.0)
		x = self._ratio*etc
		fac = 1.0-x
		np.multiply.at(loss, self._idx[self._single], fac[self._single])
		np.add.at(s, self._idx[self._xr], x[self._xr])
		loss *= 1.0-s

		gain = np.zeros(n)
		if len(self._k):
			k = self._k
			pr = np.ones(len(k))
			np.multiply.at(pr, self._row, fac[self._other])
			w = self._Pg*ep[self._i_e1]*ep[self._i_e2]
			w *= (1.0-s[k])*pr*(1.0-self._q*np.minimum(et[self._i_line[k]], 1.0))
			np.add.at(gain, k, w)
			gain[self._rows] /= self._I[self._rows]*ep[self._i_line[self._rows]]

		F, c = loss+gain, (1.0-loss)+gain
		F[self._guarded], c[self._guarded] = np.nan, np.nan
		return F, c


def _arrays(isotope, E_min=15.0, sum_tol=1.0):
	"""The `_Arrays` of a decaying state, from the decay database.

	Built at the first request and kept with the database connection, so a
	replaced data file (a re-download, another CURIE_DATA_DIR) is read anew.

	Parameters
	----------
	isotope : Isotope or str
		The decaying state, as an Isotope or any name Isotope accepts.

	E_min, sum_tol : float, optional
		Partner energy floor and summing tolerance, in keV, as in `_factors`.

	Returns
	-------
	_Arrays

	"""

	if not isinstance(isotope, Isotope):
		isotope = Isotope(isotope)
	cache = _get_connection('decay').cache.setdefault('tcs_arrays', {})
	key = (str(isotope.A)+isotope.element.upper(), isotope.isomer, float(E_min), float(sum_tol))
	if key not in cache:
		lines = isotope.gammas(xrays=True)
		xray_lines = set(lines['energy'])-set(isotope.gammas()['energy'])
		cache[key] = _Arrays(lines, _pairs(isotope), xray_lines, E_min=E_min, sum_tol=sum_tol)
	return cache[key]


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
