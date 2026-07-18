"""Tests for the DecayChain count filters and fit_config.

Covers the fit_config attribute (validator, kwarg merge, persistence), the
max_chi2 / exclude_lines / time_range / unc_R_floor filters with their drop
accounting and announcements, the p0 starting-estimate override, and the
peak-fit chi2 column get_counts carries into dc.counts.
"""
import numpy as np
import pandas as pd
import pytest

import curie as ci
from conftest import EXAMPLES_DIR, requires_data


def _synthetic_chain(chi2=None, starts=(50.0, 55.0)):
    # two well-measured counts on a 152EU chain; chi2/starts configurable
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    df = pd.DataFrame({'isotope': '152EUg', 'start': list(starts),
                       'stop': [s+0.1 for s in starts],
                       'counts': [1E5, 1E5], 'unc_counts': [400.0, 400.0],
                       'energy': [121.78, 344.28]})
    if chi2 is not None:
        df['chi2'] = chi2
    dc.counts = df
    return dc


@requires_data('decay')
def test_fit_config_validator_and_merge(capsys):
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    assert dc.fit_config['max_error'] == 0.4
    dc.fit_config = {'max_erorr': 0.3}
    out = capsys.readouterr().out
    assert "unknown key 'max_erorr'" in out and "did you mean 'max_error'" in out
    assert dc.fit_config['max_error'] == 0.4
    with pytest.raises(ValueError, match="'max_error' must be a number"):
        dc.fit_config = {'max_error': 'lots'}
    # kwargs on the fit merge and persist, like Spectrum.fit_config
    dc = _synthetic_chain()
    dc.fit_R(max_error=0.3)
    assert dc.fit_config['max_error'] == 0.3


@requires_data('decay')
def test_max_chi2_filter(capsys):
    dc = _synthetic_chain(chi2=[2.0, 50.0])
    dc.fit_R(max_chi2=25)
    out = capsys.readouterr().out
    assert '1 dropped: peak chi2>25' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 1
    assert int(dc.diagnostics['n_points'].iloc[0]) == 1
    # counts without a chi2 column are unaffected, with a warning
    dc = _synthetic_chain()
    dc.fit_R(max_chi2=25)
    out = capsys.readouterr().out
    assert 'no peak-fit chi2 column' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 0


@requires_data('decay')
def test_exclude_lines_matching(capsys):
    # closest-line match within 0.5 keV
    dc = _synthetic_chain()
    dc.fit_R(exclude_lines=[('152EU', 121.9)])
    out = capsys.readouterr().out
    assert '1 excluded: exclude_lines' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 1
    # no line inside the tolerance: warn, name the nearest, exclude nothing
    dc = _synthetic_chain()
    dc.fit_R(exclude_lines=[('152EU', 300.0)])
    out = capsys.readouterr().out
    assert 'matches no line within 0.5 keV' in out
    assert 'nearest 152EUg line is 344.3 keV' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 0
    # isotope-name form with no matching counts warns and excludes nothing
    dc = _synthetic_chain()
    dc.fit_R(exclude_lines=['154EU'])
    out = capsys.readouterr().out
    assert 'matches no counts' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 0


@requires_data('decay')
def test_time_range_filter(capsys):
    dc = _synthetic_chain()
    dc.fit_R(time_range={'152EU': (None, 52.0)})
    out = capsys.readouterr().out
    assert '1 outside time_range' in out
    assert int(dc.diagnostics['n_dropped'].iloc[0]) == 1
    assert int(dc.diagnostics['n_points'].iloc[0]) == 1


@requires_data('decay')
def test_unc_R_floor(capsys):
    dc = _synthetic_chain()
    itp, R, cov = dc.fit_R(unc_R_floor=0.5)
    out = capsys.readouterr().out
    assert 'raised to the floor 50% of R' in out
    assert np.sqrt(cov[0][0]) >= 0.5*R[0]*(1-1E-9)
    # a floor below the fitted uncertainty changes nothing
    dc = _synthetic_chain()
    itp2, R2, cov2 = dc.fit_R(unc_R_floor=1E-6)
    assert np.sqrt(cov2[0][0]) > 1E-6*R2[0]


@requires_data('decay')
def test_p0_override(capsys):
    dc = _synthetic_chain()
    itp, R_default, _ = dc.fit_R()
    dc2 = _synthetic_chain()
    ra = float(dc2.R_avg['R_avg'].iloc[0])
    itp2, R_seeded, _ = dc2.fit_R(p0={'152EU': float(R_default[0])})
    # the seed is converted from production-rate units to the fitted
    # multiplier through R_avg (the fit itself is convex, so only the stored
    # starting estimate can reveal a broken conversion)
    assert dc2._fit_result['p0'][0] == pytest.approx(float(R_default[0])/ra)
    np.testing.assert_allclose(R_seeded, R_default, rtol=1E-6)
    # unknown isotope in the seed warns and is ignored
    dc3 = _synthetic_chain()
    dc3.fit_R(p0={'60CO': 1.0})
    out = capsys.readouterr().out
    assert "p0 entry '60CO' is not a fitted isotope" in out
    with pytest.raises(ValueError, match='p0 must be a dict'):
        _synthetic_chain().fit_R(p0=3.0)


@requires_data('decay')
def test_get_counts_carries_chi2():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    assert 'chi2' in dc.counts.columns
    assert np.isfinite(dc.counts['chi2']).all()


@requires_data('decay')
def test_filters_default_off_zero_config():
    # zero-config identity: no new filter engages by default
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    itp, R, cov = dc.fit_R()
    # reference re-recorded 2026-07-18 against the v2 data generation (the
    # refreshed 152EU intensities shift the fitted rate ~0.2% and one more
    # count crosses the relative-error floor; originally recorded at the
    # doublet-merge fix on fitting-0.2.0).  The count accounting is exact
    # (a default filter engaging changes the integers); the fitted rate and
    # its relative uncertainty carry tolerances sized for cross-platform
    # optimizer drift (~1E-5 observed), far below any real change
    d = dc.diagnostics
    assert int(d['n_points'].iloc[0]) == 42 and int(d['n_dropped'].iloc[0]) == 2
    np.testing.assert_allclose(R, [27581526.896975107], rtol=1E-3)
    assert float(np.sqrt(cov[0][0])/R[0]) == pytest.approx(0.016464, rel=1E-2)
