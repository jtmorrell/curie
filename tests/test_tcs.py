"""Validation of the coincidence-summing correction factors.

The factor F of a line is the ratio of its full-energy-peak count rate with
coincidence summing to the rate without it.  The pairwise form under test
(curie._tcs) is checked against an independent oracle written here: for a
synthetic level scheme (levels, feeding fractions, branching fractions, gamma
transitions only) every decay path is enumerated, and the peak rate of a line
is summed over paths and over the photon subsets whose energies add up to the
line energy, each subset fully absorbed and every other photon of the path
escaping the detector.  The pairwise form is exact for a single cascade and
differs at second order in the efficiencies on a branched scheme.

Table-pinned cases (60Co, 22Na, 54Mn, 137Cs, 109Cd, 152Eu) use values typed
from the shipped decay database, read 2026-11-09:
decay.db sha256 aba725655dd30532b843f759348118bd9205fc9f6448a3b5fd5fbe1034d8a9dc.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

import curie as ci
from conftest import requires_data
from curie._tcs import _pairs, _factors, _summing_factor
from curie.calibration import _peak_to_total, _TAU, _SIGMA


def efficiencies(scale=1.0):
    """Closed-form peak and total efficiencies, both scaled by `scale`."""
    def eps_p(E):
        return scale * 0.08 * (np.asarray(E, dtype=float) / 100.0)**-0.9

    def eps_t(E):
        return scale * 0.15 * (np.asarray(E, dtype=float) / 100.0)**-0.3
    return eps_p, eps_t


EPS_P, EPS_T = efficiencies()


# ---------------------------------------------------------------- oracle

def decay_paths(feed, branch):
    """Every gamma path of a scheme as (probability, (transition, ...)).

    `feed` maps a level to its feeding fraction; `branch` maps a level to
    {lower level: branching fraction}.  A transition is a (high, low) tuple.
    """
    out = []

    def walk(level, prob, seq):
        if level not in branch:
            out.append((prob, tuple(seq)))
            return
        for low, b in branch[level].items():
            walk(low, prob * b, seq + [(level, low)])

    for level, f in feed.items():
        walk(level, f, [])
    return out


class Scheme(object):
    """Synthetic level scheme with its exact (enumerated) summing factors."""

    def __init__(self, levels, feed, branch, tol=1e-6):
        self.paths = decay_paths(feed, branch)
        self.levels = levels
        self.tol = tol
        trans = sorted({t for _, s in self.paths for t in s})
        self.E = {t: levels[t[0]] - levels[t[1]] for t in trans}
        self.trans = trans
        self.energies = sorted(set(self.E.values()))
        # photons per decay at each energy
        self.line = {e: sum(p * sum(1 for t in s if self.E[t] == e) for p, s in self.paths)
                     for e in self.energies}
        # coincident pairs per decay, keyed on the unordered energy pair
        self.pair = {}
        for p, s in self.paths:
            for a, b in itertools.combinations(s, 2):
                key = tuple(sorted((self.E[a], self.E[b])))
                self.pair[key] = self.pair.get(key, 0.0) + p

    def lines_frame(self):
        """One row per transition, intensities in percent (repeated energies kept)."""
        rows = [(self.E[t], 100.0 * sum(p for p, s in self.paths if t in s)) for t in self.trans]
        return pd.DataFrame(rows, columns=['energy', 'intensity'])

    def pairs_frame(self):
        """Pair rows in both orderings; an equal-energy pair stored once."""
        rows = []
        for (a, b), p in sorted(self.pair.items()):
            rows.append((a, b, 'g', 100.0 * p, np.nan))
            if a != b:
                rows.append((b, a, 'g', 100.0 * p, np.nan))
        return pd.DataFrame(rows, columns=['energy', 'coinc_energy', 'coinc_type',
                                           'intensity', 'unc_intensity'])

    def exact(self, eps_p, eps_t):
        """Exact factor and the three-or-more-photon gain T of each line energy."""
        F, T = {}, {}
        for e in self.energies:
            N, N3 = 0.0, 0.0
            for p, s in self.paths:
                En = [self.E[t] for t in s]
                for r in range(1, len(s) + 1):
                    for sub in itertools.combinations(range(len(s)), r):
                        if abs(sum(En[k] for k in sub) - e) < self.tol:
                            w = np.prod([eps_p(En[k]) for k in sub])
                            w *= np.prod([1.0 - eps_t(En[k]) for k in range(len(s)) if k not in sub])
                            N += p * w
                            if r >= 3:
                                N3 += p * w
            F[e] = N / (self.line[e] * eps_p(e))
            T[e] = N3 / (self.line[e] * eps_p(e))
        return F, T

    def bound_terms(self, eps_p, eps_t, sum_tol=1.0):
        """S, G and M of each line energy, as the second-order bound defines them."""
        S, G, M = {}, {}, {}
        for e in self.energies:
            s_sum, g_sum = 0.0, 0.0
            for (a, b), p in self.pair.items():
                if a == e or b == e:
                    partner = b if a == e else a
                    mult = 2.0 if a == b else 1.0
                    s_sum += mult * p / self.line[e] * eps_t(partner)
                if abs(a + b - e) <= sum_tol:
                    g_sum += p / self.line[e] * eps_p(a) * eps_p(b) / eps_p(e)
            S[e], G[e] = s_sum, g_sum
            m = 0.0
            for p, s in self.paths:
                En = [self.E[t] for t in s]
                if e in En:
                    others = list(En)
                    others.remove(e)
                    m = max(m, sum(eps_t(x) for x in others))
            M[e] = m
        return S, G, M

    def pairwise(self, eps_p, eps_t):
        lines = self.lines_frame()
        F = _factors(lines, self.pairs_frame(), eps_p, eps_t)
        out = {}
        for e, f in zip(lines['energy'], F):
            assert out.get(e, f) == f
            out[e] = f
        return out


LEVELS = {0: 0.0, 1: 121.8, 2: 366.5, 3: 1085.8, 4: 1529.8}

CASCADE3 = Scheme(LEVELS, {3: 1.0}, {3: {2: 1.0}, 2: {1: 1.0}, 1: {0: 1.0}})
CROSSOVER = Scheme(LEVELS, {2: 1.0}, {2: {1: 0.9, 0: 0.1}, 1: {0: 1.0}})
EQUAL = Scheme({0: 0.0, 1: 100.0, 2: 200.0}, {2: 1.0}, {2: {1: 0.8, 0: 0.2}, 1: {0: 1.0}})
BRANCHED = Scheme(LEVELS, {4: 0.6, 3: 0.3, 2: 0.1},
                  {4: {3: 0.5, 2: 0.3, 1: 0.2}, 3: {2: 0.4, 1: 0.35, 0: 0.25},
                   2: {1: 0.7, 0: 0.3}, 1: {0: 1.0}})


# ---------------------------------------------------------------- synthetic schemes

@pytest.mark.validation
@pytest.mark.parametrize('scheme', [CASCADE3, CROSSOVER, EQUAL],
                         ids=['cascade3-loss', 'crossover-gain', 'equal-energy'])
def test_pairwise_exact_on_single_cascade(scheme):
    exact, _ = scheme.exact(EPS_P, EPS_T)
    F = scheme.pairwise(EPS_P, EPS_T)
    assert sorted(F) == scheme.energies
    for e in scheme.energies:
        assert F[e] == pytest.approx(exact[e], abs=1e-12), e
    # summing moves every line of these schemes
    assert all(abs(exact[e] - 1.0) > 1e-3 for e in scheme.energies)


@pytest.mark.validation
def test_pairwise_second_order_on_branched_scheme():
    exact, T = BRANCHED.exact(EPS_P, EPS_T)
    F = BRANCHED.pairwise(EPS_P, EPS_T)
    S, G, M = BRANCHED.bound_terms(EPS_P, EPS_T)
    assert len(BRANCHED.energies) == 9
    diff = {e: F[e] - exact[e] for e in BRANCHED.energies}
    for e in BRANCHED.energies:
        assert abs(diff[e]) <= S[e] * M[e] / 2 + G[e] * M[e] + T[e], e
    nonzero = [e for e in BRANCHED.energies if abs(diff[e]) > 1e-10]
    assert nonzero
    assert max(abs(diff[e]) for e in nonzero) > 1e-4

    # halving every efficiency: the difference falls as the square of the scale
    d = {}
    for scale in (0.5, 0.25):
        ep, et = efficiencies(scale)
        ex, _ = BRANCHED.exact(ep, et)
        Fs = BRANCHED.pairwise(ep, et)
        d[scale] = {e: Fs[e] - ex[e] for e in BRANCHED.energies}
    for e in nonzero:
        assert 3.5 <= d[0.5][e] / d[0.25][e] <= 4.5, e


@pytest.mark.validation
def test_factors_empty_pairs_and_order():
    lines = pd.DataFrame({'energy': [300.0, 100.0, 300.0], 'intensity': [10.0, 50.0, 5.0]})
    pairs = pd.DataFrame(columns=['energy', 'coinc_energy', 'coinc_type', 'intensity', 'unc_intensity'])
    F = _factors(lines, pairs, EPS_P, EPS_T)
    assert len(F) == 3
    assert all(f == 1.0 for f in F)


# ---------------------------------------------------------------- decay data

def _Fsum(isotope, energy, **kw):
    return _summing_factor(isotope, energy, EPS_P, EPS_T, **kw)


@pytest.mark.library_value
@requires_data('decay')
def test_co60_hand_factors():
    F1173 = (1.0 - (99.83377701123567 / 99.85) * EPS_T(1332.492)
             + (0.006475000590701808 / 99.85) * EPS_P(347.14) * EPS_P(826.1) / EPS_P(1173.228))
    assert _Fsum('60CO', 1173.228) == pytest.approx(F1173, abs=1e-12)

    F1332 = 1.0 - (99.83377701123567 / 99.9826) * EPS_T(1173.228)
    assert _Fsum('60CO', 1332.492) == pytest.approx(F1332, abs=1e-4)

    F2505 = 1.0 + (99.83377701123567 * EPS_P(1173.228) * EPS_P(1332.492)
                   + 0.001022368514321338 * EPS_P(347.14) * EPS_P(2158.57)) / (2e-06 * EPS_P(2505.692))
    assert _Fsum('60CO', 2505.692) == pytest.approx(F2505, rel=1e-9)


@pytest.mark.library_value
@requires_data('decay')
def test_na22_annihilation_factors():
    assert _Fsum('22NA', 1274.537) == pytest.approx(1.0 - (179.802054 / 99.94) * EPS_T(511.0), abs=1e-12)
    assert _Fsum('22NA', 511.0) == pytest.approx(1.0 - (179.802054 / 179.91) * EPS_T(1274.537), abs=1e-12)


@pytest.mark.library_value
@requires_data('decay')
@pytest.mark.parametrize('isotope, energy', [('137CS', 661.657), ('109CD', 88.0336), ('54MN', 834.848)])
def test_summing_free_lines(isotope, energy):
    assert _Fsum(isotope, energy) == 1.0


@pytest.mark.library_value
@requires_data('decay')
def test_mn54_xray_partners_below_floor():
    # the only partners of the 834.848 keV line are Cr K x-rays near 5.4 keV
    assert _Fsum('54MN', 834.848, E_min=0.0) < 1.0


@pytest.mark.library_value
@requires_data('decay')
def test_eu152_close_lines_keyed_on_own_energy():
    eu = ci.Isotope('152EU')
    lines, pairs = eu.gammas(xrays=True), _pairs(eu)
    E1, E2 = 443.9606, 444.01
    rows1, rows2 = pairs[pairs['energy'] == E1], pairs[pairs['energy'] == E2]
    assert len(rows1) and len(rows2)
    assert not set(rows1.index) & set(rows2.index)

    def F_at(p):
        F = _factors(lines, p, EPS_P, EPS_T)
        return F[np.flatnonzero(lines['energy'] == E1)[0]], F[np.flatnonzero(lines['energy'] == E2)[0]]

    F1, F2 = F_at(pairs)
    assert F1 != F2
    # each factor reads only the rows keyed on its own line
    assert F_at(pairs[pairs['energy'] != E2])[0] == F1
    assert F_at(pairs[pairs['energy'] != E1])[1] == F2


@pytest.mark.library_value
@requires_data('decay')
def test_pairs_keying():
    ref = _pairs('152EU')
    assert list(ref.columns) == ['energy', 'coinc_energy', 'coinc_type', 'intensity', 'unc_intensity']
    assert len(ref) == 2229
    pd.testing.assert_frame_equal(_pairs('Eu-152'), ref)
    pd.testing.assert_frame_equal(_pairs(ci.Isotope('152EU')), ref)
    assert len(_pairs('152EUm1')) == 481
    # annihilation rows carry no uncertainty
    na = _pairs('22NA')
    assert na['unc_intensity'].isna().all()
    assert na['unc_intensity'].dtype == float
    empty = _pairs('137CS')
    assert len(empty) == 0
    assert list(empty.columns) == list(ref.columns)


@pytest.mark.library_value
@requires_data('decay')
def test_summing_factor_shapes_and_unknown_energy():
    F = _Fsum('60CO', np.array([1173.228, 1332.492]))
    assert isinstance(F, np.ndarray) and F.shape == (2,)
    assert F[0] == _Fsum('60CO', 1173.228)
    assert isinstance(_Fsum('60CO', 1173.228), float)
    with pytest.raises(ValueError, match='1173.23.*60CO'):
        _Fsum('60CO', 1173.23)


# ---------------------------------------------------------------- peak-to-total ratio

PT_ENERGIES = np.geomspace(15.0, 5000.0, 400)
PT_LENGTHS = np.geomspace(0.01, 30.0, 40)


@pytest.mark.validation
def test_peak_to_total_bounded_and_non_increasing_in_energy():
    # above about 7 MeV pair production makes the ratio rise, so the range is 15 keV-5 MeV
    for L in PT_LENGTHS:
        pt = _peak_to_total(PT_ENERGIES, L)
        assert np.all(pt > 0.0) and np.all(pt <= 1.0), L
        # a value saturated at 1 moves in the last float digit
        assert np.all(np.diff(pt) <= 1e-12), L


@pytest.mark.validation
def test_peak_to_total_increasing_in_length():
    for E in [15.0, 50.0, 122.0, 344.0, 662.0, 1408.0, 3000.0, 5000.0]:
        pt = np.array([_peak_to_total(E, L) for L in PT_LENGTHS])
        step = np.diff(pt)
        assert np.all(step >= 0.0), E
        below = pt[:-1] < 1.0 - 1e-9
        assert below.any() and np.all(step[below] > 0.0), E


@pytest.mark.validation
def test_peak_to_total_limits():
    # no Compton-scattered photon is reabsorbed in a vanishing crystal; all are in a thick one
    assert np.allclose(_peak_to_total(PT_ENERGIES, 1e-12),
                       _TAU(PT_ENERGIES) / (_TAU(PT_ENERGIES) + _SIGMA(PT_ENERGIES)), rtol=0.0, atol=1e-9)
    assert np.allclose(_peak_to_total(PT_ENERGIES, 1e4), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.characterization
def test_peak_to_total_reference_values():
    # computed 2026-11-09 from (tau + kappa*sigma)/(tau + sigma), kappa = 1 - exp(-mu*L),
    # on curie's Ge attenuation table at L = 1 cm
    for E, ref in [(122.0, 0.9481), (344.0, 0.4785), (662.0, 0.3314), (1408.0, 0.2316)]:
        assert float(_peak_to_total(E, 1.0)) == pytest.approx(ref, abs=5e-4), E
