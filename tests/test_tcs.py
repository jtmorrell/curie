"""Validation of the coincidence-summing correction factors.

The factor F of a line is the ratio of its full-energy-peak count rate with
coincidence summing to the rate without it.  The pairwise form under test
(curie._tcs) is checked against an independent oracle written here: for a
synthetic level scheme (levels, feeding fractions, branching fractions, gamma
transitions only) every decay path is enumerated, and the peak rate of a line
is summed over paths and over the photon subsets whose energies add up to the
line energy, each subset fully absorbed and every other photon of the path
escaping the detector.  The pairwise form is exact for a single cascade and
differs at second order in the efficiencies on a branched scheme.  Electron-
capture schemes add K x-rays, at most one per vacancy, from K capture and K
conversion.

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
        """S, G and M of each line energy, as the second-order bound defines them.

        A pair's summed-in term G goes only to the line nearest its summed energy.
        """
        S, G, M = {}, {}, {}
        for e in self.energies:
            s_sum, g_sum = 0.0, 0.0
            for (a, b), p in self.pair.items():
                if a == e or b == e:
                    partner = b if a == e else a
                    mult = 2.0 if a == b else 1.0
                    s_sum += mult * p / self.line[e] * eps_t(partner)
                nearest = min(self.energies, key=lambda x: abs(a + b - x))
                if nearest == e and abs(a + b - e) <= sum_tol:
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


# a crossover cascade beside a level of its own at 800.4 keV, 0.4 keV above the
# crossover line and within the default sum_tol of the 300 + 500 keV sum
CLOSE_LINE = Scheme({0: 0.0, 1: 300.0, 2: 800.0, 3: 800.4}, {2: 0.7, 3: 0.3},
                    {2: {1: 0.9, 0: 0.1}, 1: {0: 1.0}, 3: {0: 1.0}})


@pytest.mark.validation
def test_gain_counted_once_on_the_nearest_line():
    exact, _ = CLOSE_LINE.exact(EPS_P, EPS_T)
    F = CLOSE_LINE.pairwise(EPS_P, EPS_T)
    assert sorted(F) == CLOSE_LINE.energies
    assert exact[800.0] - 1.0 > 1e-3
    assert exact[800.4] == pytest.approx(1.0, abs=1e-12)
    for e in CLOSE_LINE.energies:
        assert F[e] == pytest.approx(exact[e], abs=1e-12), e


@pytest.mark.validation
def test_bound_gain_credited_to_the_nearest_line():
    # the 300 + 500 keV pair sums within sum_tol of both 800.0 and 800.4 keV; it
    # occurs in 0.7 * 0.9 decays and the 800.0 keV crossover in 0.7 * 0.1
    S, G, M = CLOSE_LINE.bound_terms(EPS_P, EPS_T)
    assert G[800.0] == pytest.approx(9.0 * EPS_P(300.0) * EPS_P(500.0) / EPS_P(800.0), rel=1e-12)
    assert G[800.4] == 0.0
    assert G[300.0] == 0.0 and G[500.0] == 0.0


# a crossover fed from above and followed by a transition below: level 4 to 3, then
# 3 to 2 (0.7) or 3 to 1 (0.3), 2 to 1, 1 to 0
CASCADE_CROSSOVER = Scheme(LEVELS, {4: 1.0}, {4: {3: 1.0}, 3: {2: 0.7, 1: 0.3}, 2: {1: 1.0}, 1: {0: 1.0}})


@pytest.mark.validation
def test_gain_carries_the_loss_inside_a_cascade():
    # the summed-in counts of the 964.0 keV crossover need the 444.0 and 121.8 keV
    # photons to escape, as its own counts do, so loss * (1 + gain) is exact on it
    sc = CASCADE_CROSSOVER
    exact, T = sc.exact(EPS_P, EPS_T)
    F = sc.pairwise(EPS_P, EPS_T)
    S, G, M = sc.bound_terms(EPS_P, EPS_T)
    assert len(sc.energies) == 5
    crossover = sc.E[(3, 1)]
    assert G[crossover] > 0.05
    # lines of one branch, the crossover among them, are exact
    for t in [(3, 1), (3, 2), (2, 1)]:
        e = sc.E[t]
        assert F[e] == pytest.approx(exact[e], abs=1e-12), e
    # lines on both branches differ at second order
    for t in [(4, 3), (1, 0)]:
        e = sc.E[t]
        assert abs(F[e] - exact[e]) <= S[e] * M[e] / 2 + G[e] * M[e] + T[e], e


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


# ---------------------------------------------------------------- K x-rays

def capture_histories(levels, feed, branch, pK, kx, conv=None):
    """Photon histories of an electron-capture scheme as (probability, ((kind, keV), ...)).

    A decay feeds a level with probability `feed[level]` by capture of a K electron
    with probability `pK`; transition (high, low) converts in the K shell with
    probability `conv[(high, low)]`, emitting no gamma ray.  A K vacancy is filled
    with emission of the K x-ray at x keV with probability `kx[x]`, at most one per
    vacancy, and of no photon otherwise.
    """
    conv = conv or {}
    out = []

    def vacancy(prob, photons):
        alt = [(prob * q, photons + (('x', x),)) for x, q in kx.items()]
        return alt + [(prob * (1.0 - sum(kx.values())), photons)]

    def walk(level, prob, photons):
        if level not in branch:
            out.append((prob, photons))
            return
        for low, b in branch[level].items():
            a = conv.get((level, low), 0.0)
            walk(low, prob * b * (1.0 - a), photons + (('g', levels[level] - levels[low]),))
            if a > 0.0:
                for pr, ph in vacancy(prob * b * a, photons):
                    walk(low, pr, ph)

    for level, f in feed.items():
        for pr, ph in [(f * (1.0 - pK), ())] + vacancy(f * pK, ()):
            walk(level, pr, ph)
    return out


class PhotonScheme(object):
    """Gamma rays and K x-rays of a set of decay histories, with exact summing factors.

    The pair table follows the decay database: gamma-gamma pairs 'g' (an
    equal-energy pair stored once), gamma plus x-ray pairs 'XR', no x-ray plus
    x-ray pairs.
    """

    def __init__(self, histories):
        self.histories = histories
        self.line, self.kind, self.pair = {}, {}, {}
        for p, ph in histories:
            for k, e in ph:
                self.line[e] = self.line.get(e, 0.0) + p
                self.kind[e] = k
            for (ka, a), (kb, b) in itertools.combinations(ph, 2):
                if ka == 'x' and kb == 'x':
                    continue
                key = ('XR' if 'x' in (ka, kb) else 'g',) + tuple(sorted((a, b)))
                self.pair[key] = self.pair.get(key, 0.0) + p
        self.energies = sorted(self.line)
        self.xray_lines = [e for e in self.energies if self.kind[e] == 'x']

    def pairs_frame(self):
        rows = []
        for (t, a, b), p in sorted(self.pair.items()):
            rows.append((a, b, t, 100.0 * p, np.nan))
            if a != b:
                rows.append((b, a, t, 100.0 * p, np.nan))
        return pd.DataFrame(rows, columns=['energy', 'coinc_energy', 'coinc_type',
                                           'intensity', 'unc_intensity'])

    def pairwise(self, eps_p, eps_t):
        lines = pd.DataFrame({'energy': self.energies,
                              'intensity': [100.0 * self.line[e] for e in self.energies]})
        F = _factors(lines, self.pairs_frame(), eps_p, eps_t, xray_lines=self.xray_lines)
        return dict(zip(self.energies, F))

    def exact(self, eps_p, eps_t, tol=1e-6):
        """Exact factor and three-or-more-photon gain T of each line energy."""
        F, T = {}, {}
        for e in self.energies:
            N, N3 = 0.0, 0.0
            for p, ph in self.histories:
                En = [x for _, x in ph]
                for r in range(1, len(En) + 1):
                    for sub in itertools.combinations(range(len(En)), r):
                        if abs(sum(En[k] for k in sub) - e) < tol:
                            w = np.prod([eps_p(En[k]) for k in sub])
                            w *= np.prod([1.0 - eps_t(En[k]) for k in range(len(En)) if k not in sub])
                            N += p * w
                            if r >= 3:
                                N3 += p * w
            F[e] = N / (self.line[e] * eps_p(e))
            T[e] = N3 / (self.line[e] * eps_p(e))
        return F, T

    def bound_terms(self, eps_p, eps_t, sum_tol=1.0):
        """S, G and M of each line energy, as `Scheme.bound_terms` defines them."""
        S, G, M = {}, {}, {}
        for e in self.energies:
            s_sum, g_sum = 0.0, 0.0
            for (t, a, b), p in self.pair.items():
                if a == e or b == e:
                    partner = b if a == e else a
                    mult = 2.0 if a == b else 1.0
                    s_sum += mult * p / self.line[e] * eps_t(partner)
                nearest = min(self.energies, key=lambda x: abs(a + b - x))
                if t == 'g' and nearest == e and abs(a + b - e) <= sum_tol:
                    g_sum += p / self.line[e] * eps_p(a) * eps_p(b) / eps_p(e)
            S[e], G[e] = s_sum, g_sum
            m = 0.0
            for p, ph in self.histories:
                En = [x for _, x in ph]
                if e in En:
                    En.remove(e)
                    m = max(m, sum(eps_t(x) for x in En))
            M[e] = m
        return S, G, M


# K x-rays of one vacancy (Sm-like energies; emission probabilities sum to the
# fluorescence yield 0.93), above 35.1 keV, where EPS_P stays below EPS_T
KX = {39.522: 0.25, 40.117: 0.45, 45.523: 0.18, 46.575: 0.05}

# one K vacancy per decay: K capture (0.85) feeds 700 keV, which decays by a
# 400 + 300 keV cascade (0.8) or the crossover (0.2); no x-ray sum lies near a line
ONE_VACANCY = PhotonScheme(capture_histories({0: 0.0, 1: 300.0, 2: 700.0}, {2: 1.0},
                                             {2: {1: 0.8, 0: 0.2}, 1: {0: 1.0}}, 0.85, KX))
# two K vacancies: K capture, then K conversion of the 250 or 150 keV transition
TWO_VACANCY = PhotonScheme(capture_histories({0: 0.0, 1: 150.0, 2: 400.0}, {2: 1.0},
                                             {2: {1: 0.9, 0: 0.1}, 1: {0: 1.0}}, 0.85, KX,
                                             conv={(2, 1): 0.3, (1, 0): 0.15}))


@pytest.mark.validation
def test_xray_partners_of_one_vacancy_exact():
    sc = ONE_VACANCY
    exact, T = sc.exact(EPS_P, EPS_T)
    F = sc.pairwise(EPS_P, EPS_T)
    S, G, M = sc.bound_terms(EPS_P, EPS_T)
    assert sc.energies == sorted(KX) + [300.0, 400.0, 700.0]
    assert G[700.0] > 0.05
    for e in [300.0, 400.0, 700.0]:
        assert any(t == 'XR' and e in (a, b) for t, a, b in sc.pair), e
        assert F[e] == pytest.approx(exact[e], abs=1e-12), e
    # an x-ray line sits on both gamma branches: second order
    for e in KX:
        assert abs(F[e] - exact[e]) <= S[e] * M[e] / 2 + G[e] * M[e] + T[e], e


@pytest.mark.validation
def test_xray_partners_of_two_vacancies_second_order():
    sc = TWO_VACANCY
    exact, T = sc.exact(EPS_P, EPS_T)
    F = sc.pairwise(EPS_P, EPS_T)
    S, G, M = sc.bound_terms(EPS_P, EPS_T)
    # two x-rays in one decay
    assert any(sum(k == 'x' for k, _ in ph) == 2 for _, ph in sc.histories)
    # the gamma lines, whose x-ray partners share one factor; an x-ray line misses
    # the x-ray of the other vacancy at first order, as the table pairs no two x-rays
    gammas = [150.0, 250.0, 400.0]
    diff = {e: F[e] - exact[e] for e in gammas}
    for e in gammas:
        assert abs(diff[e]) <= S[e] * M[e] / 2 + G[e] * M[e] + T[e], e
    assert max(abs(diff[e]) for e in gammas) > 1e-4


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
    F1173 = (1.0 - (99.83377701123567 / 99.85) * EPS_T(1332.492)) * (
        1.0 + (0.006475000590701808 / 99.85) * EPS_P(347.14) * EPS_P(826.1) / EPS_P(1173.228))
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
