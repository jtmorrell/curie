"""Render tests (Agg backend) for the fit-visibility plot upgrades.

Every surface follows the same rule: a fit's plot never hides evidence.
DecayChain.plot draws a 1-sigma band from the stored fit covariance and shows
fit/plot-excluded counts as open grey markers; Spectrum.plot marks failed
multiplets; the Calibration plots show rejected points as open red markers.
The band is exact (the activity is linear in the fitted multipliers), checked
here against the fitted covariance analytically.
"""
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import numpy as np
import pandas as pd
import pytest

import curie as ci
from conftest import EXAMPLES_DIR, requires_data

EU_SOURCES = [{'isotope': '152EU', 'A0': 3.7E4, 'ref_date': '01/01/2009 12:00:00'}]


def _legend_texts(ax):
    lg = ax.get_legend()
    return [t.get_text() for t in lg.get_texts()] if lg else []


@requires_data('decay')
def test_decay_plot_band_and_excluded_markers():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    dc.fit_R()  # default max_error drops 2 counts -> grey markers
    f, ax = plt.subplots()
    dc.plot(f=f, ax=ax, show=False)
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    texts = _legend_texts(ax)
    assert r'fit $\pm 1\sigma$' in texts
    assert 'excluded from fit/plot' in texts
    plt.close(f)


@requires_data('decay')
def test_band_analytic_single_isotope():
    # single-isotope fit_A0: at t=0 the band equals the fitted A0 uncertainty
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    itp, A, cov_norm = dc.fit_A0()
    sig = dc._band_sigma(itp[0], np.array([0.0]))
    np.testing.assert_allclose(sig[0], np.sqrt(cov_norm[0][0]), rtol=1E-9)
    # and the relative band width is constant for a single-isotope chain
    t = np.array([0.0, 20.0, 100.0])
    rel = dc._band_sigma(itp[0], t)/dc.activity(itp[0], t)
    np.testing.assert_allclose(rel, rel[0], rtol=1E-9)


@requires_data('decay')
def test_band_absent_without_fit():
    dc = ci.DecayChain('99MO', A0=3.5E8, units='d')
    f, ax = plt.subplots()
    dc.plot(f=f, ax=ax, show=False)
    assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close(f)


def test_spectrum_plot_marks_failed_multiplet(capsys):
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    # negative forward-fit amplitude -> invalid bounds -> failed multiplet
    sp.fit_peaks(gammas=[{'energy': 500.0, 'intensity': 100.0, 'unc_intensity': 0.1, 'isotope': '154EU'}], SNR_min=-1E9)
    f, ax = plt.subplots()
    sp.plot(f=f, ax=ax, show=False)
    # red hatched fill of the unfitted counts, announced by a warning - no
    # legend entry
    out = capsys.readouterr().out
    assert '1 failed multiplet peak fits - indicated in the plot with red hatching' in out
    hatched = [c for c in ax.collections if isinstance(c, PolyCollection) and c.get_hatch()]
    assert len(hatched) == 1
    assert 'failed fit' not in _legend_texts(ax)
    plt.close(f)


@requires_data('decay')
def test_calibration_plots_show_rejected_points():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES)
    # the eu calibration rejects rescal and effcal points but no engcal points
    for name, expect in [('plot_engcal', False), ('plot_rescal', True), ('plot_effcal', True)]:
        f, ax = plt.subplots()
        getattr(cb, name)(f=f, ax=ax, show=False)
        assert ('rejected points' in _legend_texts(ax)) == expect, name
        plt.close(f)
