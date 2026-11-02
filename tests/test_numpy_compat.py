"""Coverage for the two public features whose quadrature must work on numpy 1.x and 2.x.

`Isotope.dose_rate` and the rebinning branch of `Spectrum.__add__` integrate with the
trapezoid rule. numpy renamed its trapezoid function in 2.0 and removed the old name in 2.4,
so these lines break on an import-clean install unless they are exercised directly.
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data

# Frozen reference values recorded 2026-11-02 at curie commit 261832f under numpy 2.2.6, in
# R/hr at activity=3.7E10 Bq and distance=30 cm, with the curie-data v2 decay.db. Expected to
# change on a decay-data rebuild - re-record deliberately. An unexpected shift at any other
# time is a regression in the range integrals. These detect change; they are not independent
# truth. The 60CO total is the 1576.6518 R/hr of the dose_rate docstring. 60CO reaches the
# electron range function, 22NA the positron range function, and 241AM the alpha range function.
DOSE_REFERENCE = {
    '60CO': {
        'alphas': 0.0,
        'beta_minus': 1545.1657548480377,
        'beta_plus': 0.0,
        'electrons': 17.032022398386985,
        'gammas': 14.454039953270152,
        'total': 1576.651817199695,
    },
    '22NA': {
        'alphas': 0.0,
        'beta_minus': 0.0,
        'beta_plus': 849.0941823942469,
        'electrons': 3048.069871134579,
        'gammas': 12.660274442754513,
        'total': 3909.8243279715803,
    },
    '241AM': {
        'alphas': 306561.95615840505,
        'beta_minus': 0.0,
        'beta_plus': 0.0,
        'electrons': 8144.477864508169,
        'gammas': 0.16402082359382253,
        'total': 314706.5980437368,
    },
}
DOSE_NONZERO_TERM = {'60CO': 'beta_minus', '22NA': 'beta_plus', '241AM': 'alphas'}


@pytest.mark.library_value
@requires_data('decay')
@pytest.mark.parametrize('isotope', sorted(DOSE_REFERENCE))
def test_dose_rate_components_frozen_reference(isotope):
    dose = ci.Isotope(isotope).dose_rate(activity=3.7E10, distance=30.0, units='R/hr')
    assert float(dose[DOSE_NONZERO_TERM[isotope]]) > 0.0
    assert sorted(dose) == sorted(DOSE_REFERENCE[isotope])
    for component, value in DOSE_REFERENCE[isotope].items():
        assert float(dose[component]) == pytest.approx(value, rel=1E-9), component


N_CHANNELS = 4096
SELF_ENGCAL = [0.0, 0.5]    # keV, keV/channel
OTHER_ENGCAL = [1.0, 0.4]
PEAK_CHANNEL = 1500         # in `other`: 1.0 + 0.4*1500 = 601 keV
PEAK_SIGMA = 3.0
PEAK_AMPLITUDE = 1.0E8      # high counts make the Poisson limits below about 0.02% on the total
SELF_PEAK_CHANNEL = 500


def _gaussian_spectrum(channel, engcal, start_time):
    # A Gaussian peak with no continuum: the histogram is exactly zero outside the peak.
    # `__add__` drops counts outside `self`'s energy range, and a peak well inside it loses none.
    x = np.arange(N_CHANNELS)
    sp = ci.Spectrum()
    sp.hist = (PEAK_AMPLITUDE*np.exp(-0.5*((x - channel)/PEAK_SIGMA)**2)).astype(np.int64)
    sp.cb.engcal = np.array(engcal, dtype=np.float64)
    sp.start_time, sp.live_time, sp.real_time = start_time, 100.0, 101.0
    return sp


@pytest.mark.validation
def test_add_rebins_onto_self_calibration():
    a = _gaussian_spectrum(SELF_PEAK_CHANNEL, SELF_ENGCAL, 'self')
    b = _gaussian_spectrum(PEAK_CHANNEL, OTHER_ENGCAL, 'other')
    self_counts, other_counts = a.hist.copy(), b.hist.copy()

    np.random.seed(1234)
    added = (a + b).hist - self_counts

    # the mismatched calibrations move the peak to (1.0 + 0.4*1500)/0.5 = 1202 in `self`
    assert np.all(added >= 0)
    assert np.argmax(added) == 1202

    # Each `self` channel receives a Poisson draw whose mean is the integrated counts truncated
    # to an integer, which removes less than one count per channel from the mean. The channels
    # that can lose counts are those the peak in `other` spans once rebinned.
    N = other_counts.sum()
    support = other_counts.nonzero()[0]
    span = np.ceil((support[-1] - support[0] + 1)*OTHER_ENGCAL[1]/SELF_ENGCAL[1]) + 2
    limit = 5.0*np.sqrt(N)
    assert N - span - limit < added.sum() < N + limit

    # Position and width below the one-channel resolution of argmax. The variance in `self`
    # channels sums the peak's rebinned width, the linear interpolation between `other`'s channels
    # (spacing h: h^2/6), a `self` channel's width (1/12), and the 10-point trapezoid rule over it.
    channels = np.arange(N_CHANNELS)
    total = added.sum()
    centroid = (channels*added).sum()/total
    width = np.sqrt(((channels - centroid)**2*added).sum()/total)
    h = OTHER_ENGCAL[1]/SELF_ENGCAL[1]
    expected_width = np.sqrt((PEAK_SIGMA*h)**2 + h**2/6.0 + 1.0/12.0 + (1.0/9.0)**2/6.0)
    assert abs(centroid - 1202.0) < 5.0*expected_width/np.sqrt(total)
    assert abs(width - expected_width) < 5.0*expected_width/np.sqrt(2.0*total)
