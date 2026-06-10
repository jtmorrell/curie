"""Bateman-equation validation tests (public suite, category 1).

Truth source: independent closed-form solutions implemented in this module, with decay
constants and branching ratios read from `ci.Isotope` at runtime. These tests therefore
validate the DecayChain solver in isolation and are unaffected by nuclear-data library
rebuilds; only the explicitly marked `library_value` test pins a data value.

Time-axis convention (verified on curie 0.0.34): with an R production history,
DecayChain.activity(time=t) measures t from end of bombardment (EoB); with A0 initial
activities, t is measured from the reference time of A0.
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data

pytestmark = [pytest.mark.validation, requires_data('decay')]


def _lam(isotope, units):
    return ci.Isotope(isotope).decay_const(units)


@pytest.mark.parametrize('t', [0.0, 0.5, 13.537, 50.0])
def test_single_isotope_activity_exponential(t):
    # 152EU: both decay branches terminate at stable nuclides, so A(t) = A0*exp(-lm*t)
    A0 = 3.7E4
    dc = ci.DecayChain('152EU', A0=A0, units='y')
    expected = A0 * np.exp(-_lam('152EU', 'y') * t)
    assert float(dc.activity('152EU', time=t)) == pytest.approx(expected, rel=1E-9)


@pytest.mark.parametrize('interval', [(0.0, 1.0), (1.0, 2.0), (0.0, 1.0E3)])
def test_single_isotope_decays_integral(interval):
    # decays(t1, t2) = integral of A dt = (A0/lm_s)*(exp(-lm*t1) - exp(-lm*t2)),
    # with the 1/lm in absolute seconds since activity is in Bq
    t1, t2 = interval
    A0 = 3.7E3
    dc = ci.DecayChain('152EU', A0=A0, units='h')
    lm_h, lm_s = _lam('152EU', 'h'), _lam('152EU', 's')
    expected = A0 * (np.exp(-lm_h * t1) - np.exp(-lm_h * t2)) / lm_s
    assert float(dc.decays('152EU', t_start=t1, t_stop=t2)) == pytest.approx(expected, rel=1E-9)


@pytest.mark.parametrize('t', [1.0, 10.0, 66.0, 200.0])
def test_branching_chain_99mo_99mtc(t):
    # Two-member Bateman with branching:
    # A_d(t) = BR * A0_p * (lm_d/(lm_d-lm_p)) * (exp(-lm_p*t) - exp(-lm_d*t))
    A0 = 1.0E6
    dc = ci.DecayChain('99MO', A0=A0, units='h')
    lm_p, lm_d = _lam('99MO', 'h'), _lam('99TCm1', 'h')
    br = ci.Isotope('99MO').decay_products['99TCm1']
    expected = br * A0 * (lm_d / (lm_d - lm_p)) * (np.exp(-lm_p * t) - np.exp(-lm_d * t))
    assert float(dc.activity('99TCm1', time=t)) == pytest.approx(expected, rel=1E-12)


class TestIngrowth134Ce:
    """Constant production rate R of 134CE for T days, then decay.

    134CE -> 134LA is pure EC (BR = 1), so the daughter reference solution is the clean
    two-member form. With production source br*lm_p*N_p(s), N_p(s) = (R/lm_p)(1-exp(-lm_p*s)):

      N_d(T)     = br*R*[ (1-exp(-lm_d*T))/lm_d - (exp(-lm_p*T)-exp(-lm_d*T))/(lm_d-lm_p) ]
      N_d(T+tau) = N_p(T)*br*lm_p*(exp(-lm_p*tau)-exp(-lm_d*tau))/(lm_d-lm_p)
                   + N_d(T)*exp(-lm_d*tau)
    """
    R, T = 2.5E5, 2.0  # production rate [1/d], irradiation length [d]

    @pytest.fixture(scope='class')
    def chain(self):
        return ci.DecayChain('134CE', R={'134CE': [[self.R, self.T]]}, units='d')

    @pytest.mark.parametrize('tau', [0.0, 1.0, 5.0])
    def test_parent_saturation(self, chain, tau):
        lm_p = _lam('134CE', 'd')
        expected = self.R * (1.0 - np.exp(-lm_p * self.T)) * np.exp(-lm_p * tau)
        assert float(chain.activity('134CE', time=tau)) == pytest.approx(expected, rel=1E-12)

    @pytest.mark.parametrize('tau', [0.0, 0.01, 1.0, 5.0])
    def test_daughter_ingrowth(self, chain, tau):
        lm_p, lm_d = _lam('134CE', 'd'), _lam('134LA', 'd')
        br = ci.Isotope('134CE').decay_products['134LAg']
        N_p_T = (self.R / lm_p) * (1.0 - np.exp(-lm_p * self.T))
        N_d_T = br * self.R * ((1.0 - np.exp(-lm_d * self.T)) / lm_d
                               - (np.exp(-lm_p * self.T) - np.exp(-lm_d * self.T)) / (lm_d - lm_p))
        N_d = (N_p_T * br * lm_p * (np.exp(-lm_p * tau) - np.exp(-lm_d * tau)) / (lm_d - lm_p)
               + N_d_T * np.exp(-lm_d * tau))
        assert float(chain.activity('134LA', time=tau)) == pytest.approx(lm_d * N_d, rel=1E-12)


def test_secular_equilibrium_137cs():
    # T1/2 ratio ~6E6 (30.08 y vs 2.552 min): after a few daughter half-lives the
    # activity ratio A(137BAm1)/A(137CS) equals the branching ratio
    dc = ci.DecayChain('137CS', A0=1.0, units='y')
    br = ci.Isotope('137CS').decay_products['137BAm1']
    t = 1.0E-3  # ~8.8 h: >> daughter T1/2, << parent T1/2
    ratio = float(dc.activity('137BAm1', time=t)) / float(dc.activity('137CS', time=t))
    assert ratio == pytest.approx(br, rel=1E-4)


@pytest.mark.library_value
def test_half_life_152eu_frozen_reference():
    """Frozen reference value, recorded 2026-06-09 from decay.db (NuDat 2.0 era), curie
    v0.0.34. Expected to change at the Stage-6 decay-data rebuild - re-record deliberately."""
    assert ci.Isotope('152EU').half_life('y') == pytest.approx(13.537, rel=1E-4)
