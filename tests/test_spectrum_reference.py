"""Spectrum + Calibration reference-value tests (public suite, category 3).

Characterization against frozen reference values recorded 2026-06-09 from curie v0.0.34
(scipy 1.17.1, numpy 2.2.6, pandas 2.3.2) on examples/eu_calib_7cm.Spe - the 152Eu
sealed-source spectrum at 7 cm that ships with the package. The reference is a recorded
run, not an independent truth: an unexpected failure means fitting/calibration behavior
changed - investigate, and re-record deliberately only for an intended change.

Only counts and efficiency-curve values are pinned - never unc_* columns (a planned
rework of the uncertainty model is expected to change those).

The source activity (A0 = 3.7E4 Bq, ref 01/01/2009 12:00:00) follows the shipped
spectroscopy_examples.py and is nominal, not certificate-traceable: the efficiency tests
pin curve shape and fit stability, not absolute detector efficiency.
"""
import numpy as np
import pytest

import curie as ci
from conftest import EXAMPLES_DIR, requires_data

pytestmark = [pytest.mark.characterization, requires_data('ziegler', 'decay')]

# net peak counts, recorded 2026-06-09, curie v0.0.34
REFERENCE_SINGLES = {
    121.7817: 498785.0,
    244.6974: 80193.2,
    344.2785: 213990.0,
    778.9045: 46800.3,
    1112.0760: 34719.0,
    1408.0130: 43377.8,
}
# the 152EU 1457.64 / 40K 1460.82 doublet, fit as one multiplet
REFERENCE_MULTIPLET = {1457.643: ('152EU', 1270.5), 1460.820: ('40K', 2985.8)}
REFERENCE_N_PEAKS = 45
# efficiency curve from the documented calibrate() workflow, evaluated at reference energies
REFERENCE_EFF = {
    121.7817: 0.033881,
    344.2785: 0.015169,
    778.9045: 0.0070933,
    1408.0130: 0.0040089,
}
REFERENCE_ENGCAL_SLOPE = 0.182647  # keV/channel


def _peak(pks, energy):
    rows = pks[np.isclose(pks['energy'], energy, atol=0.1)]
    assert len(rows) == 1, 'expected exactly one peak at {} keV, got {}'.format(energy, len(rows))
    return rows.iloc[0]


@pytest.mark.parametrize('energy', sorted(REFERENCE_SINGLES))
def test_reference_peak_counts(eu_spectrum, energy):
    # 2% tolerance: the fit is deterministic in a fixed environment; the margin absorbs
    # scipy/numpy version-to-version optimizer drift without masking real changes
    pk = _peak(eu_spectrum.peaks, energy)
    assert pk['isotope'] == '152EU'
    assert pk['counts'] == pytest.approx(REFERENCE_SINGLES[energy], rel=0.02)


def test_multiplet_decomposition(eu_spectrum):
    # the 1457.6/1460.8 doublet must be decomposed into both components; component areas
    # get 5% (they trade against each other within the joint fit)
    pks = eu_spectrum.peaks
    components = {E: _peak(pks, E) for E in REFERENCE_MULTIPLET}
    for E, (isotope, counts) in REFERENCE_MULTIPLET.items():
        assert components[E]['isotope'] == isotope
        assert components[E]['counts'] == pytest.approx(counts, rel=0.05)
    # both components come from the same joint multiplet fit
    assert components[1457.643]['chi2'] == components[1460.820]['chi2']


def test_peak_identification(eu_spectrum):
    pks = eu_spectrum.peaks
    assert set(pks['isotope']) == {'152EU', '40K'}
    for energy in list(REFERENCE_SINGLES) + list(REFERENCE_MULTIPLET):
        _peak(pks, energy)  # asserts exactly one match each
    assert len(pks) == REFERENCE_N_PEAKS


def test_efficiency_calibration():
    # the documented workflow: calibrate against the (nominal) source definition from
    # spectroscopy_examples.py, on a fresh Spectrum so the session fixture stays unmodified
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    cb = ci.Calibration()
    cb.calibrate([sp], sources=[{'isotope': '152EU', 'A0': 3.7E4,
                                 'ref_date': '01/01/2009 12:00:00'}])

    # energy calibration is the most stable fitted quantity
    assert cb.engcal[1] == pytest.approx(REFERENCE_ENGCAL_SLOPE, rel=1E-3)

    # efficiency curve pinned at the reference energies (3%), not by fit parameters -
    # the 5-parameter effcal is strongly correlated and parameter-wise pins would be brittle
    for energy, eff in REFERENCE_EFF.items():
        assert float(cb.eff(energy)) == pytest.approx(eff, rel=0.03)

    # physical shape: positive everywhere, monotone decreasing above the knee
    grid = np.linspace(50.0, 2000.0, 500)
    assert np.all(cb.eff(grid) > 0.0)
    above_knee = np.linspace(150.0, 2000.0, 200)
    assert np.all(np.diff(cb.eff(above_knee)) < 0.0)
