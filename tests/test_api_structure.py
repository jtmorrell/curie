"""Structural tests of the public API (public suite, category 5).

One test per public class, mirroring the documented (docstring) happy path and asserting
structure, units, and physical invariants - never exact data values. These hold across
nuclear-data library rebuilds; verbatim docstring execution (doctests) is planned once
the known docstring errors are fixed.
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data

pytestmark = requires_data('ziegler')  # import-level requirement; per-test needs added below


@requires_data('decay')
def test_isotope_structure():
    ip = ci.Isotope('60CO')
    # unit conversion consistency (1 y = 365.25 d in curie's conventions)
    assert ip.half_life('s') > 0.0
    assert ip.half_life('d') == pytest.approx(ip.half_life('y') * 365.25, rel=1E-12)
    # naming-convention equivalence: El-AAA and AAAEL forms resolve identically
    assert ci.Isotope('Co-60').name == ip.name
    gm = ip.gammas()
    assert {'energy', 'intensity', 'unc_intensity'} <= set(gm.columns)
    assert np.all(gm['energy'] > 0.0) and np.all(gm['energy'] < 4000.0)  # keV
    assert np.all(gm['intensity'] > 0.0)
    assert isinstance(ip.decay_products, dict)
    assert all(0.0 < br <= 1.0 for br in ip.decay_products.values())


@requires_data('IRDFF')
def test_reaction_structure():
    rx = ci.Reaction('115IN(n,g)', 'irdff')
    assert rx.name.startswith('115IN(n,g)')
    assert isinstance(rx.eng, np.ndarray) and isinstance(rx.xs, np.ndarray)
    assert len(rx.eng) == len(rx.xs) > 0
    assert np.all(rx.eng > 0.0) and np.all(rx.eng <= 200.0)  # MeV
    assert np.all(rx.xs >= 0.0)                              # mb
    assert float(rx.interpolate(1.0)) >= 0.0


@requires_data('endf')
def test_reaction_name_parsing():
    # the docstring notation variants must parse and resolve
    rx = ci.Reaction('235U(n,f)', 'endf')
    assert len(rx.eng) > 0
    assert np.all(rx.xs >= 0.0)


@requires_data('decay')
def test_decay_chain_structure():
    dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
    assert len(dc.isotopes) > 0
    assert all(isinstance(ip, str) for ip in dc.isotopes)
    assert dc.isotopes[0] == '152EUg'
    assert float(dc.activity('152EU', time=0.0)) == pytest.approx(3.7E3, rel=1E-12)
    act = dc.activity('152EU', time=np.array([0.0, 1.0, 2.0]))
    assert isinstance(act, np.ndarray) and len(act) == 3


@requires_data('decay')
def test_spectrum_structure(eu_spectrum):
    sp = eu_spectrum
    assert isinstance(sp.hist, np.ndarray) and len(sp.hist) == 16384
    assert 0.0 < sp.live_time <= sp.real_time
    assert isinstance(sp.cb, ci.Calibration)
    documented = {'isotope', 'energy', 'counts', 'unc_counts', 'intensity', 'efficiency',
                  'decays', 'unc_decays', 'decay_rate', 'unc_decay_rate', 'chi2',
                  'start_time', 'live_time', 'real_time'}
    assert documented <= set(sp.peaks.columns)


def test_calibration_structure(tmp_path):
    cb = ci.Calibration()
    assert len(cb.engcal) >= 2
    assert len(cb.rescal) == 2
    assert len(cb.effcal) in (5, 7)
    # energy <-> channel round trip within half a channel
    cb.engcal = [0.0, 0.5]
    ch = cb.map_channel(661.7)
    assert abs(float(cb.eng(ch)) - 661.7) <= 0.25
    # json save/load round trip preserves all calibrations
    fn = str(tmp_path / 'calib.json')
    cb.saveas(fn)
    cb2 = ci.Calibration(fn)
    assert np.allclose(cb2.engcal, cb.engcal)
    assert np.allclose(cb2.effcal, cb.effcal)
    assert np.allclose(cb2.rescal, cb.rescal)


@requires_data('decay')
def test_element_structure():
    el = ci.Element('Fe')
    assert el.Z == 26
    assert el.mass == pytest.approx(55.85, abs=0.2)
    assert el.density > 0.0
    # S(density=1E-3) is documented as the mass stopping power: linear = mass * rho(mg/cm3)
    S_lin = float(el.S(20.0))                     # MeV/cm
    S_mass = float(el.S(20.0, density=1E-3))      # MeV/(mg/cm2)
    assert S_lin == pytest.approx(S_mass * el.density * 1E3, rel=1E-9)
    assert {'isotope', 'abundance', 'unc_abundance'} <= set(el.abundances.columns)
    assert float(el.abundances['abundance'].sum()) == pytest.approx(100.0, abs=0.1)


@requires_data('decay')
def test_compound_structure():
    cm = ci.Compound('H2O')   # chemical-formula path
    assert set(cm.weights['element']) == {'H', 'O'}
    assert {'element', 'Z', 'atom_weight', 'mass_weight'} <= set(cm.weights.columns)
    assert 'SS_316' in ci.COMPOUND_LIST           # preset path
    ss = ci.Compound('SS_316')
    assert ss.density > 0.0
    assert len(ss.weights) > 1


@requires_data('endf')
def test_library_structure():
    lb = ci.Library('endf')
    assert lb.name == 'ENDF/B-VIII.1'
    assert lb.search(target='59CO', incident='n', outgoing='g') == ['59CO(n,g)60CO']
    q = np.asarray(lb.retrieve(target='59CO', incident='n', outgoing='g'))
    assert q.ndim == 2 and q.shape[1] == 2 and q.shape[0] > 0
    assert np.all(q[:, 0] > 0.0)   # MeV
    assert np.all(q[:, 1] >= 0.0)  # mb


@requires_data('decay')
def test_stack_structure():
    stack_def = [{'name': 'Al01', 'compound': 'Al', 'thickness': 1.5},
                 {'name': 'Cu01', 'compound': 'Cu', 'thickness': 0.5}]
    st = ci.Stack(stack_def, E0=30.0, particle='p')
    assert {'name', 'compound', 'areal_density', 'mu_E', 'sig_E'} <= set(st.stack.columns)
    # areal density from thickness: 1E2 * rho(g/cm3) * t(mm) -> mg/cm2
    al = ci.Element('Al')
    assert float(st.stack.iloc[0]['areal_density']) == pytest.approx(1E2 * al.density * 1.5, rel=1E-6)
    # energy degrades monotonically through the stack, with finite straggling widths
    assert np.all(np.diff(st.stack['mu_E']) < 0.0)
    assert np.all(st.stack['mu_E'] > 0.0) and np.all(st.stack['mu_E'] < 30.0)
    assert np.all(st.stack['sig_E'] > 0.0)
    eng, flux = st.get_flux('Cu01')
    assert len(eng) == len(flux) > 0
    assert np.all(flux >= 0.0)
    assert np.all((eng > 0.0) & (eng < 30.0))
