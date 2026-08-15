"""Known-truth validation of the constrained same-isotope doublet fit.

Builds a synthetic spectrum containing the 133Ba 79.6142 / 80.9979 keV pair - two
gammas of one isotope 1.3837 keV apart, at the ~12.4:1 emission-intensity ratio of
the shipped 133Ba library (79.6142 keV at 2.6495%, 80.9979 keV at 32.9486%). The
two peaks are placed at a coarse HPGe binning (0.383 keV/channel) and detector
resolution (0.90-channel width) where they sit ~4 resolution widths apart: close
enough that a joint fit with two independent widths lets the weaker line's width
drift outward and absorb area from the stronger line's low-energy tail.

The peaks are generated from curie's own peak shape (with a realistic low-energy
tail), so the emitted area of each line is known exactly, and the fit models the
tail (skew_fit), so the reported counts are unbiased when the fit recovers the true
shape. The constrained fit ties the pair to a single shared width and the known
energy separation; against the unconstrained fit the weaker line's area is
recovered rather than inflated.
"""
import datetime as dtm

import numpy as np
import pytest

import curie as ci
from conftest import requires_data

pytestmark = [pytest.mark.validation, requires_data('decay')]

# 133Ba line energies (keV) and shipped-library emission intensities (percent)
E_WEAK, E_STRONG = 79.6142, 80.9979
I_WEAK, I_STRONG = 2.649535, 32.948550

# synthetic detector: coarse binning and resolution placing the pair ~4 widths apart
KEV_PER_CH = 0.383
SIG_CH = 0.90
# generated peak shape: a low-energy tail stronger than a bare Gaussian, as real HPGe
# peaks carry; both lines share the shape so the intensity ratio is preserved
R_GEN, ALPHA_GEN = 0.3, 1.5
STRONG_AMP = 1.6e5     # amplitude of the strong line; weak scales by the intensity ratio
CONTINUUM = 3.0e3      # flat background counts per channel
N_CHANNELS = 300       # short spectrum: only the 79.6/81.0 doublet is in range
SEED = 24              # fixed Poisson realization


def _area(amp):
    # net counts of one generated peak: Gaussian core plus the low-energy skew tail,
    # matching curie's counts accounting for a skew-fit peak
    return 2.506628*amp*SIG_CH + 2*amp*R_GEN*ALPHA_GEN*SIG_CH*np.exp(-0.5/ALPHA_GEN**2)


def _synthetic_doublet():
    ch = np.arange(N_CHANNELS).astype(float)
    mu_weak, mu_strong = E_WEAK/KEV_PER_CH, E_STRONG/KEV_PER_CH
    weak_amp = STRONG_AMP*(I_WEAK/I_STRONG)

    shape = ci.Spectrum()
    shape.hist = np.zeros(N_CHANNELS, dtype=np.int64)
    shape.cb.engcal = [0.0, KEV_PER_CH, 0.0]
    shape.cb.rescal = [SIG_CH, 0.0]
    lam = (shape._peak(ch, weak_amp, mu_weak, SIG_CH, R_GEN, ALPHA_GEN, 0.0)
           + shape._peak(ch, STRONG_AMP, mu_strong, SIG_CH, R_GEN, ALPHA_GEN, 0.0)
           + CONTINUUM)
    hist = np.random.default_rng(SEED).poisson(lam).astype(np.int64)

    sp = ci.Spectrum()
    sp.hist = hist
    sp.cb.engcal = [0.0, KEV_PER_CH, 0.0]
    sp.cb.rescal = [SIG_CH, 0.0]
    sp.start_time = dtm.datetime(2020, 1, 1)
    sp.live_time = sp.real_time = 3600.0
    sp.fit_config = {'skew_fit': True}
    sp._snip_bg()
    sp.isotopes = ['133BA']
    return sp, _area(weak_amp), _area(STRONG_AMP)


def test_constrained_doublet_recovers_areas():
    sp, truth_weak, truth_strong = _synthetic_doublet()
    pks = sp.peaks

    weak = pks[np.isclose(pks['energy'], E_WEAK, atol=0.1)]
    strong = pks[np.isclose(pks['energy'], E_STRONG, atol=0.1)]
    assert len(weak) == 1 and len(strong) == 1, 'both doublet lines must be fit'
    weak, strong = weak.iloc[0], strong.iloc[0]

    # each fitted area is consistent with its known emitted area
    assert abs(weak['counts']-truth_weak) < 3*weak['unc_counts']
    assert abs(strong['counts']-truth_strong) < 3*strong['unc_counts']

    # the weak line, prone to width inflation beside the much stronger line, is
    # recovered to within 10% of truth rather than absorbing the stronger line's area
    assert abs(weak['counts']-truth_weak)/truth_weak < 0.10
