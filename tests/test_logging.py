"""Tests for the logging spine and shared config validation.

Covers the console messages (emitted and suppressed per level), the ci-level
API (set_log_level / quiet_warnings / log_to), the fit_config validator, and
the precise ValueErrors that replaced formerly opaque failures. Messages are
asserted through capsys - the console handler writes to the current
sys.stdout, so capture behaves exactly as it does for print().
"""
import os
import re

import numpy as np
import pytest

import curie as ci
from curie import _log as clog
from conftest import EXAMPLES_DIR, requires_data


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    ci.set_log_level('INFO')
    for handler in list(clog._file_handlers.values()):
        clog._root.removeHandler(handler)
        handler.close()
    clog._file_handlers.clear()


@pytest.fixture()
def eu_refit(eu_spectrum):
    """A fresh fit_peaks call on the shared eu spectrum (fixture fit happened
    before capture starts)."""
    def run():
        return eu_spectrum.fit_peaks()
    return run


@pytest.fixture(scope='module')
def eu_local():
    """A module-private eu spectrum for tests that mutate state: calibrate()
    writes the fitted calibration back into its spectra, which must not leak
    into the session-scoped eu_spectrum fixture."""
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    sp.fit_peaks()
    return sp


########################
# Level API and handlers
########################

@requires_data('decay')
def test_info_summary_emitted(eu_refit, capsys):
    eu_refit()
    out = capsys.readouterr().out
    assert '[INFO] Spectrum(eu_calib_7cm.Spe).fit_peaks: fit ' in out
    assert 'dropped' in out
    assert 'SNR<' in out


@requires_data('decay')
def test_quiet_warnings_silences_console(eu_refit, capsys):
    ci.quiet_warnings()
    eu_refit()
    assert capsys.readouterr().out == ''


@requires_data('decay')
def test_warning_level_hides_info(eu_refit, capsys):
    ci.set_log_level('WARNING')
    eu_refit()
    out = capsys.readouterr().out
    assert '[INFO]' not in out


@requires_data('decay')
def test_debug_level_shows_per_drop_lines(eu_refit, capsys):
    ci.set_log_level('DEBUG')
    eu_refit()
    out = capsys.readouterr().out
    assert '[DEBUG] Spectrum(eu_calib_7cm.Spe).fit_peaks: dropped ' in out
    assert 'SNR_min' in out or 'I_min' in out


def test_set_log_level_rejects_unknown():
    with pytest.raises(ValueError, match='Unknown log level'):
        ci.set_log_level('CHATTY')


def test_log_to_overwrites_by_default(tmp_path):
    log_file = str(tmp_path / 'run.log')
    ci.log_to(log_file)
    clog._get_logger('spectrum').info('first run line')
    ci.log_to(log_file)
    clog._get_logger('spectrum').info('second run line')
    with open(log_file) as f:
        content = f.read()
    assert 'second run line' in content
    assert 'first run line' not in content


def test_log_to_append_mode(tmp_path):
    log_file = str(tmp_path / 'run.log')
    ci.log_to(log_file)
    clog._get_logger('spectrum').info('first run line')
    ci.log_to(log_file, mode='append')
    clog._get_logger('spectrum').info('second run line')
    with open(log_file) as f:
        content = f.read()
    assert 'first run line' in content
    assert 'second run line' in content


def test_log_to_accepts_open_style_modes(tmp_path):
    ci.log_to(str(tmp_path / 'w.log'), mode='w')
    ci.log_to(str(tmp_path / 'a.log'), mode='a')
    with pytest.raises(ValueError, match="mode must be 'overwrite' or 'append'"):
        ci.log_to(str(tmp_path / 'x.log'), mode='x')


def test_log_file_level_independent_of_console(tmp_path, capsys):
    log_file = str(tmp_path / 'debug.log')
    ci.quiet_warnings()
    ci.log_to(log_file, level='DEBUG')
    clog._get_logger('spectrum').debug('detail line')
    assert capsys.readouterr().out == ''
    with open(log_file) as f:
        assert 'detail line' in f.read()


########################
# Config validation
########################

def test_unknown_fit_config_key_warns_with_suggestion(capsys):
    sp = ci.Spectrum()
    sp.fit_config = {'SNR_Min': 5}
    out = capsys.readouterr().out
    assert "[WARNING] Spectrum.fit_config: unknown key 'SNR_Min' ignored" in out
    assert "did you mean 'SNR_min'?" in out
    assert 'SNR_Min' not in sp.fit_config
    assert sp.fit_config['SNR_min'] == 4.0


def test_unknown_fit_config_key_without_close_match(capsys):
    sp = ci.Spectrum()
    sp.fit_config = {'zzqx': 1}
    out = capsys.readouterr().out
    assert "unknown key 'zzqx' ignored" in out
    assert 'did you mean' not in out


def test_fit_config_type_error():
    sp = ci.Spectrum()
    with pytest.raises(ValueError, match="'SNR_min' must be a number"):
        sp.fit_config = {'SNR_min': 'four'}


def test_fit_config_choice_error():
    sp = ci.Spectrum()
    with pytest.raises(ValueError, match="'bg' must be one of"):
        sp.fit_config = {'bg': 'cubic'}


def test_fit_config_valid_keys_accepted():
    sp = ci.Spectrum()
    sp.fit_config = {'SNR_min': 5.0, 'bg': 'quadratic', 'xrays': True, 'E_min': (50.0, 2000.0)}
    assert sp.fit_config['SNR_min'] == 5.0
    assert sp.fit_config['bg'] == 'quadratic'


def test_fit_config_computed_values_accepted():
    # values computed from numpy expressions are legitimate config inputs:
    # np.bool_ for BOOLEAN keys, integer-valued floats for INTEGER keys
    sp = ci.Spectrum()
    sp.fit_config = {'xrays': np.bool_(True), 'skew_fit': np.array([True]).any(),
                     'multi_max': 8.0, 'I_min': np.float64(0.05)}
    assert sp.fit_config['xrays']
    assert sp.fit_config['multi_max'] == 8.0
    with pytest.raises(ValueError, match="'multi_max' must be an integer"):
        sp.fit_config = {'multi_max': 8.5}


@requires_data('decay')
def test_numpy_scalar_intensity_bound_dispatch():
    # a numpy scalar passes NUMBER_OR_PAIR validation and must be dispatched
    # as a scalar bound, not indexed as a (lower, upper) pair
    g = ci.Isotope('152EU').gammas(I_lim=np.float64(1.0))
    assert len(g) and (g['intensity'] >= 1.0).all()
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    sp.fit_config = {'I_min': np.float64(0.05)}
    assert sp.fit_peaks() is not None


@requires_data('decay')
def test_intensity_range_drop_message(capsys):
    # a (lower, upper) I_min renders as a range, not as 'intensity < (tuple)%'
    ci.set_log_level('DEBUG')
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    sp.fit_config = {'I_min': (1.0, 20.0)}
    sp.fit_peaks()
    out = capsys.readouterr().out
    assert 'outside I_min [1.0, 20.0]%' in out
    assert 'intensity outside [1.0, 20.0]%' in out
    assert '< I_min (1.0' not in out


########################
# Precise errors at formerly opaque sites
########################

@requires_data('decay')
def test_fit_R_requires_counts():
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    with pytest.raises(ValueError, match='fit_R requires count data'):
        dc.fit_R()


@requires_data('decay')
def test_fit_A0_requires_counts():
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    with pytest.raises(ValueError, match='fit_A0 requires count data'):
        dc.fit_A0()


@requires_data('decay')
def test_fit_R_all_counts_filtered():
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.counts = [[50.0, 50.1, 1000.0, 900.0]]  # 90% relative error
    with pytest.raises(ValueError, match=r'0 of 1 counts pass the filters \(1 relative error>40%'):
        dc.fit_R()


@requires_data('decay')
def test_get_counts_bad_EoB():
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    with pytest.raises(ValueError, match='could not parse EoB'):
        dc.get_counts([], EoB='13/45/2016 08:39:08')


@requires_data('decay')
def test_get_counts_no_matches(eu_spectrum):
    dc = ci.DecayChain('64CU', A0=3.7E4, units='d')
    with pytest.raises(ValueError, match='no counts found'):
        dc.get_counts([eu_spectrum], EoB='01/01/2016 08:39:08')


@requires_data('ziegler', 'decay')
def test_calibrate_no_matching_sources(eu_local):
    cb = ci.Calibration()
    with pytest.raises(ValueError, match='no calibration points'):
        cb.calibrate([eu_local], sources=[{'isotope': '60CO', 'A0': 3.7E4,
                                              'ref_date': '01/01/2009 12:00:00'}])


@requires_data('ziegler', 'decay')
def test_calibrate_bad_ref_date(eu_local):
    cb = ci.Calibration()
    with pytest.raises(ValueError, match="could not parse ref_date"):
        cb.calibrate([eu_local], sources=[{'isotope': '152EU', 'A0': 3.7E4,
                                              'ref_date': '2009-01-01'}])


def test_download_unknown_db():
    with pytest.raises(ValueError, match="is not a curie database"):
        ci.download('not_a_database')


########################
# Fit announcements on real data
########################

@requires_data('ziegler', 'decay')
def test_calibrate_summary_lines(eu_local, capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_local], sources=[{'isotope': '152EU', 'A0': 3.7E4,
                                          'ref_date': '01/01/2009 12:00:00'}])
    out = capsys.readouterr().out
    assert '[INFO] Calibration.calibrate: engcal [' in out
    assert '[INFO] Calibration.calibrate: rescal [' in out
    assert '[INFO] Calibration.calibrate: effcal [vidmar-' in out
    assert 'chi2/dof=' in out
    # the effcal line reports the pre-inflation chi2/dof: whenever the scale
    # factor fired, the reported goodness of fit must still show the
    # inconsistency (the converged whitened chi2 tends to 1 by construction)
    line = next(l for l in out.splitlines() if 'effcal [vidmar-' in l)
    if 'scaled x' in line:
        chi2 = float(re.search(r'chi2/dof=([0-9.eE+-]+)', line).group(1))
        assert chi2 > 1.0


@requires_data('ziegler', 'decay')
def test_fit_R_summary_line(capsys):
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
    dc.get_counts([sp], EoB='01/01/2016 08:39:08')
    dc.fit_R()
    out = capsys.readouterr().out
    assert '[INFO] DecayChain(152EUg).get_counts: ' in out
    assert ' counts from 1 of 1 spectra for ' in out
    assert '[INFO] DecayChain(152EUg).fit_R: fit 1 production rates to ' in out
    assert 'chi2/dof=' in out


@requires_data('decay')
def test_scale_factor_folded_into_summary(capsys):
    # mutually inconsistent counts: the inflation is announced on the summary
    # line itself, not as a separate message
    dc = ci.DecayChain('152EU', A0=3.7E4, units='d')
    dc.counts = [[0.1*n, 0.1*n+0.1, c, 10.0] for n, c in
                 enumerate([1000.0, 500.0, 800.0, 900.0, 700.0, 600.0])]
    dc.fit_A0()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if 'initial activities' in l and 'fit ' in l)
    assert 'chi2/dof=' in line and 'scaled x' in line
    assert 'mutually inconsistent' not in out
    assert '[INFO] DecayChain(152EUg).fit_A0: ' in line


@requires_data('decay')
def test_empty_fit_announces_once(capsys):
    # an empty fit re-runs on every sp.peaks access; the announcement fires
    # once, repeats at DEBUG only, and returns after a config change
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU']
    sp.fit_config = {'E_min': 1E5}
    sp.fit_peaks()
    assert capsys.readouterr().out.count('nothing to fit') == 1
    sp.peaks
    sp.peaks
    assert 'nothing to fit' not in capsys.readouterr().out
    sp.fit_config = {'E_min': 2E5}
    sp.peaks
    assert capsys.readouterr().out.count('nothing to fit') == 1


@requires_data('decay')
def test_zero_config_results_unchanged(eu_spectrum):
    # the logging spine must not perturb results: identical fits at any level
    ci.set_log_level('DEBUG')
    pks_debug = eu_spectrum.fit_peaks()
    ci.quiet_warnings()
    pks_quiet = eu_spectrum.fit_peaks()
    assert np.allclose(pks_debug['counts'].to_numpy(), pks_quiet['counts'].to_numpy())
    assert len(pks_debug) == len(pks_quiet)
