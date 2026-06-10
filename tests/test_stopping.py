"""Stopping-power / range / attenuation validation tests (public suite, category 2).

External anchors: NIST PSTAR/ASTAR CSDA ranges and the Hubbell-Seltzer photon mass
attenuation tabulation. Curie implements the Anderson-Ziegler (AZ) stopping-power
formulation, not ICRU-49 (the PSTAR/ASTAR basis), so the charged-particle tolerances are
set ~2-3x above the agreement measured on curie v0.0.34 (2026-06-09):

    protons in Al/Cu  : within +/-0.6%  -> tol 1.5%
    protons in H2O    : within 3.1%     -> tol 5%   (AZ Bragg additivity vs ICRU-49 water)
    alphas in Al      : within +/-0.8%  -> tol 2%
    photons in Pb     : exact           -> tol 0.5% (ziegler.db photon data IS the
                                                     Hubbell-Seltzer tabulation)

These catch implementation regressions, not methodology differences.

Anchor provenance (all retrieved 2026-06-09):
- PSTAR/ASTAR CSDA ranges (g/cm2): NIST, physics.nist.gov/PhysRefData/Star. The NIST CGI is
  POST-only, so values were taken from two independent mirrors in exact mutual agreement
  (github.com/jacyap/RangeCalc_ProtonsElectrons text tables; github.com/Zelenyy/
  nist-calculators NIST_STAR.hdf5), cross-checked against the PBTWiki water table that
  quotes NIST PSTAR directly.
- Photon mass attenuation mu/rho (cm2/g), Pb: NIST XAAMDI (Hubbell & Seltzer),
  physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html (static table, fetched direct).

The internal-consistency and Stack tests are anchor-free: they compare curie against direct
quadrature of curie's own S(E), so they hold under any stopping-power data change.

NB: Element construction reads natural abundances from decay.db, so even these tests
require ziegler+decay (relevant to the minimal-CI data fixture).
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data

pytestmark = requires_data('ziegler', 'decay')

# NIST PSTAR CSDA range [g/cm2] for protons
PSTAR_CSDA = {
    'Al':  {10.0: 0.1705, 30.0: 1.180, 100.0: 10.01},
    'Cu':  {10.0: 0.2199, 30.0: 1.446, 100.0: 11.85},
}
PSTAR_CSDA_H2O = {10.0: 0.1230, 30.0: 0.8853, 100.0: 7.718}

# NIST ASTAR CSDA range [g/cm2] for helium ions in Al
ASTAR_CSDA_AL = {10.0: 0.016659, 20.0: 0.052158, 40.0: 0.17119}

# NIST XAAMDI mu/rho [cm2/g] for Pb (without coherent-scattering exclusion)
XCOM_MU_PB = {1000.0: 7.102E-02, 1250.0: 5.876E-02, 1500.0: 5.222E-02}


@pytest.mark.validation
@pytest.mark.parametrize('element,energy', [(el, E) for el in PSTAR_CSDA for E in PSTAR_CSDA[el]])
def test_proton_range_vs_pstar(element, energy):
    el = ci.Element(element)
    csda = float(el.range(energy)) * el.density  # range() returns cm at natural density
    assert csda == pytest.approx(PSTAR_CSDA[element][energy], rel=0.015)


@pytest.mark.validation
@pytest.mark.parametrize('energy', sorted(PSTAR_CSDA_H2O))
def test_proton_range_water_vs_pstar(energy):
    cm = ci.Compound('H2O', density=1.0)
    csda = float(cm.range(energy, density=1.0)) * 1.0
    assert csda == pytest.approx(PSTAR_CSDA_H2O[energy], rel=0.05)


@pytest.mark.validation
@pytest.mark.parametrize('energy', sorted(ASTAR_CSDA_AL))
def test_alpha_range_vs_astar(energy):
    el = ci.Element('Al')
    csda = float(el.range(energy, particle='a')) * el.density
    assert csda == pytest.approx(ASTAR_CSDA_AL[energy], rel=0.02)


@pytest.mark.validation
@pytest.mark.parametrize('energy', sorted(XCOM_MU_PB))
def test_photon_attenuation_vs_hubbell(energy):
    pb = ci.Element('Pb')
    assert float(pb.mu(energy)) == pytest.approx(XCOM_MU_PB[energy], rel=0.005)


@pytest.mark.validation
@pytest.mark.parametrize('element', ['Al', 'Cu'])
def test_range_is_integral_of_inverse_S(element):
    # Anchor-free: range(E2) - range(E1) must equal the quadrature of dE/S over [E1, E2]
    # using curie's own stopping power, whatever data it carries.
    el = ci.Element(element)
    E1, E2 = 20.0, 80.0
    grid = np.linspace(E1, E2, 4001)
    S = el.S(grid)                              # MeV/cm at natural density
    quad = np.trapezoid(1.0 / S, grid)          # cm
    delta_range = float(el.range(E2) - el.range(E1))
    assert delta_range == pytest.approx(quad, rel=5E-3)


@pytest.mark.validation
@pytest.mark.parametrize('particle', ['p', 'a'])
def test_range_monotonic_positive(particle):
    el = ci.Element('Al')
    grid = np.linspace(2.0, 150.0, 100)
    r = el.range(grid, particle=particle)
    assert np.all(r > 0.0)
    assert np.all(np.diff(r) > 0.0)


@pytest.mark.characterization
def test_stack_endpoint_self_consistency():
    # 30 MeV protons through Al then Cu; compare the solver's mean foil-exit energy with
    # direct CSDA stepping of the same S(E). Characterization: same physics inputs, no
    # independent truth - it pins the transport solver's behavior.
    stack_def = [{'name': 'Al01', 'compound': 'Al', 'thickness': 1.5},   # mm
                 {'name': 'Cu01', 'compound': 'Cu', 'thickness': 0.5}]
    st = ci.Stack(stack_def, E0=30.0, particle='p')

    def csda_exit(E_in, compound, ad_mg_cm2):
        # thickness traversed while slowing from E_in to E: t(E) = integral_E^Ein dE'/S(E');
        # invert t(E) at the foil areal density to get the exit energy
        from scipy.integrate import cumulative_trapezoid
        cm = ci.Compound(compound)
        grid = np.linspace(1.0, E_in, 5000)
        S = cm.S(grid, density=1E-3)         # mass stopping power, MeV/(mg/cm2)
        cum = cumulative_trapezoid(1.0 / S, grid, initial=0.0)
        t_at_E = cum[-1] - cum               # mg/cm2, decreasing in E
        return float(np.interp(ad_mg_cm2, t_at_E[::-1], grid[::-1]))

    # mu_E is the mean proton energy IN each foil; compare it to the CSDA energy at the
    # foil midpoint (half the areal density), then step the full foil to enter the next.
    E = 30.0
    for _, row in st.stack.iterrows():
        E_mid = csda_exit(E, row['compound'], 0.5 * row['areal_density'])
        mu_E = float(st.stack[st.stack['name'] == row['name']]['mu_E'].iloc[0])
        # measured 2026-06-09 (v0.0.34): -0.19% (Al), -0.41% (Cu) - straggling/flux
        # weighting vs the midpoint approximation; 1% gives ~2.5x headroom
        assert mu_E == pytest.approx(E_mid, rel=0.01), row['name']
        E = csda_exit(E, row['compound'], row['areal_density'])
