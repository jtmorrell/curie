"""Tests for the .diagnostics tables and the public fit-data surfaces.

Covers the shared schema (identical core columns on Spectrum, Calibration and
DecayChain; empty-with-schema before any fit), population on the real
eu_calib_7cm.Spe fit, flag detection on constructed at-bound and failed fits,
the tidy cb.*_data point tables (used and dropped points, .json round-trip,
old-format files), sp.fits, dc._fit_result, and that reading any of these
surfaces never triggers or alters a fit.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import curie as ci
from conftest import EXAMPLES_DIR, requires_data

CORE_COLUMNS = ['fit', 'chi2', 'dof', 'n_points', 'n_dropped', 'converged',
                'model', 'scale_factor', 'flags', 'message']

EU_SOURCES = [{'isotope': '152EU', 'A0': 3.7E4, 'ref_date': '01/01/2009 12:00:00'}]


@pytest.fixture(scope='module')
def eu_calibrated():
    """A locally calibrated eu spectrum + Calibration (calibrate() mutates its
    spectra, so the session eu_spectrum fixture must not be used here)."""
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES)
    return sp, cb


########################
# Schema
########################

def test_empty_schema_before_fit():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    cb = ci.Calibration()
    assert sp.diagnostics.columns.tolist() == CORE_COLUMNS+['energy_min', 'energy_max', 'isotopes', 'n_peaks']
    assert cb.diagnostics.columns.tolist() == CORE_COLUMNS
    assert len(sp.diagnostics) == 0
    assert len(cb.diagnostics) == 0
    # reading diagnostics must not have triggered a fit
    assert sp._fits is None and sp._peaks is None
    assert sp.fits is None


@requires_data('decay')
def test_empty_schema_decay_chain():
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    assert dc.diagnostics.columns.tolist() == CORE_COLUMNS+['isotope']
    assert len(dc.diagnostics) == 0
    assert dc._fit_result is None


########################
# Spectrum
########################

@requires_data('decay')
def test_spectrum_diagnostics_populated(eu_spectrum):
    d = eu_spectrum.diagnostics
    assert len(d) == len(eu_spectrum.fits)
    assert d['fit'].tolist() == ['multiplet {0}'.format(i+1) for i in range(len(d))]
    assert d['converged'].all()
    assert (d['model'] == 'snip').all()
    assert (d['n_points'] > 0).all() and (d['dof'] > 0).all()
    assert (d['energy_max'] > d['energy_min']).all()
    assert (d['n_peaks'] >= 1).all()
    assert d['isotopes'].str.contains('152EU').any()
    # scale factor is sqrt(chi2) above 1, exactly 1 otherwise
    hi = d['chi2'] > 1.0
    assert np.allclose(d.loc[hi, 'scale_factor'], np.sqrt(d.loc[hi, 'chi2']))
    assert (d.loc[~hi, 'scale_factor'] == 1.0).all()
    # chi2_high flag tracks the fixed reporting threshold
    assert ((d['chi2'] > 10.0) == d['flags'].str.contains('chi2_high')).all()


@requires_data('decay')
def test_spectrum_diagnostics_no_refit_or_alteration(eu_spectrum):
    before = eu_spectrum.peaks.copy()
    fits_id = id(eu_spectrum._fits)
    d1, d2 = eu_spectrum.diagnostics, eu_spectrum.diagnostics
    d1.loc[:, 'flags'] = 'clobbered'
    pd.testing.assert_frame_equal(before, eu_spectrum.peaks)
    assert id(eu_spectrum._fits) == fits_id
    assert not (eu_spectrum.diagnostics['flags'] == 'clobbered').any()
    assert d2 is not d1


@requires_data('decay')
def test_spectrum_diagnostics_reset_on_config_change(eu_spectrum):
    assert len(eu_spectrum.diagnostics)
    eu_spectrum.fit_config = {'SNR_min': eu_spectrum.fit_config['SNR_min']}
    assert len(eu_spectrum.diagnostics) == 0
    eu_spectrum.fit_peaks()
    assert len(eu_spectrum.diagnostics)


def test_spectrum_at_bound_flag(capsys):
    # cap the amplitude bound far below the true 121.8 keV peak height: the
    # fitted amplitude must end on its upper bound
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    gammas = [{'energy': 121.78, 'intensity': 28.6, 'unc_intensity': 0.1, 'isotope': '152EU'}]
    sp.fit_peaks(gammas=gammas, A_bound=0.001)
    d = sp.diagnostics
    assert len(d) == 1
    assert 'at_bound:A' in d['flags'].iloc[0]
    assert "parameter 'A' at upper bound for 152EU 121.8 keV (flag: at_bound:A)" in d['message'].iloc[0]
    out = capsys.readouterr().out
    # per-peak detail stays at DEBUG; the summary carries the count
    assert '1 peaks with parameters at fit bounds (see sp.diagnostics)' in out
    assert 'at upper bound for' not in out


def test_spectrum_fit_failed_row():
    # a candidate whose forward-fit amplitude is negative produces invalid
    # bounds, so the multiplet fit fails; the diagnostics row must survive
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    gammas = [{'energy': 500.0, 'intensity': 100.0, 'unc_intensity': 0.1, 'isotope': '154EU'}]
    sp.fit_peaks(gammas=gammas, SNR_min=-1E9)
    d = sp.diagnostics
    assert len(d) == 1
    assert not d['converged'].iloc[0]
    assert d['flags'].iloc[0] == 'fit_failed'
    assert 'peak fit failed' in d['message'].iloc[0]
    assert np.isnan(d['chi2'].iloc[0])
    assert sp.fits == []


@requires_data('decay')
def test_spectrum_fits_public(eu_spectrum):
    fits = eu_spectrum.fits
    assert fits is eu_spectrum._fits
    for key in ['l', 'h', 'p0', 'bounds', 'istp', 'df', 'fit', 'unc']:
        assert key in fits[0]


@requires_data('decay')
def test_summary_points_at_diagnostics(eu_spectrum, capsys):
    eu_spectrum.fit_peaks()
    out = capsys.readouterr().out
    if 'chi2/dof>10' in out or 'at fit bounds' in out:
        assert '(see sp.diagnostics)' in out


@requires_data('decay')
def test_summary_counts_match_diagnostics_flags(eu_spectrum, capsys):
    # the console counts and the diagnostics flags come from the same loop,
    # so they must agree (a degenerate fit with non-finite chi2 is excluded
    # from both - it surfaces through its NaN chi2 instead)
    eu_spectrum.fit_peaks()
    out = capsys.readouterr().out
    d = eu_spectrum.diagnostics
    m = re.search(r'(\d+) multiplets with chi2/dof>10', out)
    assert (int(m.group(1)) if m else 0) == int(d['flags'].str.contains('chi2_high').sum())


########################
# Calibration
########################

@requires_data('decay')
def test_calibration_diagnostics_rows(eu_calibrated):
    _, cb = eu_calibrated
    d = cb.diagnostics
    assert d['fit'].tolist() == ['engcal', 'rescal', 'effcal']
    assert d['converged'].all()
    assert d['model'].iloc[0] in ('linear', 'quadratic')
    assert d['model'].iloc[1] in ('linear', 'sqrt')
    assert d['model'].iloc[2] in ('vidmar-5', 'vidmar-7')
    assert (d['n_points'] > 0).all()
    assert (d['scale_factor'] >= 1.0).all()
    assert d['message'].str.contains('fit to').all()


@requires_data('decay')
def test_calibration_data_tables(eu_calibrated):
    _, cb = eu_calibrated
    for name, cols in [('engcal_data', ['channel', 'energy', 'unc_channel']),
                       ('rescal_data', ['channel', 'width', 'unc_width']),
                       ('effcal_data', ['energy', 'efficiency', 'unc_efficiency', 'isotope'])]:
        t = getattr(cb, name)
        assert t.columns.tolist() == cols+['used', 'reason', 'residual']
        assert t['used'].any()
        assert (t.loc[t['used'], 'reason'] == '').all()
        assert (t.loc[~t['used'], 'reason'] != '').all()
        assert np.isfinite(t['residual']).all()
    # the tidy table carries every point: used plus dropped equals the total
    d = cb.diagnostics.set_index('fit')
    assert len(cb.rescal_data) == d.loc['rescal', 'n_points']+int(np.sum(cb.rescal_data['reason'] == 'unc>33%'))
    # provenance rides with every efficiency point; the correlation-group
    # line label stays in the private storage the covariance grouping reads
    assert (cb.effcal_data['isotope'] == '152EU').all()
    assert 'line' not in cb.effcal_data.columns
    assert all(':' in ln for ln in cb._calib_data['effcal']['line'])


@requires_data('decay')
def test_calibration_classic_groups_used_only(eu_calibrated):
    # raw readers of the classic groups must keep seeing used points only
    _, cb = eu_calibrated
    for grp, n_col in [('engcal', 'channel'), ('rescal', 'channel'), ('effcal', 'energy')]:
        t = getattr(cb, grp+'_data')
        assert len(cb._calib_data[grp][n_col]) == int(t['used'].sum())
        if grp+'_dropped' in cb._calib_data:
            assert len(cb._calib_data[grp+'_dropped'][n_col]) == int((~t['used']).sum())


@requires_data('decay')
def test_calibration_json_round_trip(eu_calibrated, tmp_path):
    _, cb = eu_calibrated
    f = str(tmp_path / 'cb.json')
    cb.saveas(f)
    cb2 = ci.Calibration(f)
    for name in ['engcal_data', 'rescal_data', 'effcal_data']:
        pd.testing.assert_frame_equal(getattr(cb, name), getattr(cb2, name))
    # a save file written by an older curie (no dropped groups, no provenance)
    # must load with used-only rows and blank reasons
    js = json.load(open(f))
    for grp in ['engcal_dropped', 'rescal_dropped', 'effcal_dropped']:
        js['_calib_data'].pop(grp, None)
    for key in ['isotope', 'line']:
        js['_calib_data']['effcal'].pop(key, None)
    old = str(tmp_path / 'cb_old.json')
    json.dump(js, open(old, 'w'))
    cb3 = ci.Calibration(old)
    assert cb3.engcal_data['used'].all()
    assert (cb3.engcal_data['reason'] == '').all()
    assert (cb3.effcal_data['isotope'] == '').all()
    np.testing.assert_allclose(cb3.effcal_data['residual'], cb.effcal_data[cb.effcal_data['used']]['residual'])


def test_calibration_data_empty_without_fit():
    cb = ci.Calibration()
    for name in ['engcal_data', 'rescal_data', 'effcal_data']:
        assert len(getattr(cb, name)) == 0
        assert 'used' in getattr(cb, name).columns


########################
# DecayChain
########################

@requires_data('decay')
def test_decay_chain_diagnostics_and_fit_result():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    itp, R, cov = dc.fit_R()
    d = dc.diagnostics
    assert len(d) == len(itp)
    assert (d['fit'] == 'fit_R').all()
    assert d['isotope'].tolist() == list(itp)
    assert d['chi2'].nunique() == 1
    assert (d['n_points']+d['n_dropped'] == len(dc.counts)).all()
    assert (d['scale_factor'] >= 1.0).all()
    assert d['message'].str.contains('production rates').all()
    fr = dc._fit_result
    assert fr['label'] == 'fit_R' and fr['isotopes'] == list(itp)
    np.testing.assert_array_equal(fr['cov_norm'], cov)
    np.testing.assert_array_equal(fr['value'], R)
    assert fr['dof'] == int(d['dof'].iloc[0])


@requires_data('decay')
def test_decay_chain_fit_a0_diagnostics():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    itp, A, cov = dc.fit_A0()
    d = dc.diagnostics
    assert (d['fit'] == 'fit_A0').all()
    assert d['isotope'].tolist() == list(itp)
    assert dc._fit_result['label'] == 'fit_A0'
    np.testing.assert_array_equal(dc._fit_result['cov_norm'], cov)


@requires_data('decay')
def test_fit_a0_singular_covariance_fallback(capsys):
    # a singular covariance from the GLS fit must be announced and replaced
    # by the same finite fallback fit_R applies - not returned as NaN/inf
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    real = dc._gls_fit
    def singular_gls(*args, **kwargs):
        fit, cov, chi2, scale, note = real(*args, **kwargs)
        return fit, np.full(cov.shape, np.inf), chi2, scale, note
    dc._gls_fit = singular_gls
    itp, A, cov_norm = dc.fit_A0()
    out = capsys.readouterr().out
    assert 'covariance estimate is singular' in out
    assert '(flag: singular_cov)' in out
    assert np.all(np.isfinite(cov_norm))
    d = dc.diagnostics
    assert d['flags'].str.contains('singular_cov').all()
    assert d['message'].str.contains('flag: singular_cov', regex=False).all()
