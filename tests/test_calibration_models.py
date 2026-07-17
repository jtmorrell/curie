"""Tests for the selectable calibration models and Calibration.fit_config.

Covers the model tags (explicit state, .json persistence, infer-by-length
backcompat), the loglog efficiency model (whose parameter length can collide
with Vidmar's, making the tag load-bearing), the cubic energy calibration and
its numeric inverse, the sqrt_quad resolution model, the eff_points
augment-and-refit, the threshold knobs, and the extrapolation warning.
"""
import json

import numpy as np
import pandas as pd
import pytest

import curie as ci
import curie.calibration as cal_mod
from conftest import EXAMPLES_DIR, requires_data

EU_SOURCES = [{'isotope': '152EU', 'A0': 3.7E4, 'ref_date': '01/01/2009 12:00:00'}]


def eu_sp():
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    return sp


def test_fit_config_validator(capsys):
    cb = ci.Calibration()
    cb.fit_config = {'effcal_modle': 'loglog'}
    out = capsys.readouterr().out
    assert "unknown key 'effcal_modle'" in out and "did you mean 'effcal_model'" in out
    with pytest.raises(ValueError, match="'effcal_model' must be one of"):
        cb.fit_config = {'effcal_model': 'spline'}
    with pytest.raises(ValueError, match="'effcal_model' must be one of"):
        cb.fit_config = {'effcal_model': 'loglog-12'}


@requires_data('decay')
def test_zero_config_identity():
    # curve values recorded at the doublet-merge fix on fitting-0.2.0 (the
    # 443.96/444.01 pair now merges, correcting a 9x-wrong efficiency point
    # and moving the whole curve by several percent) - a change means a
    # default moved.  The efficiency CURVE is pinned rather than the raw
    # parameters: the effective-length parameter is nearly degenerate once
    # saturated, so different scipy/BLAS stacks converge to different
    # parameterizations of the same curve (observed cross-platform curve
    # reproducibility ~2E-6; tolerance gives 50x margin)
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES)
    e = np.array([121.7817, 344.2785, 778.9045, 1408.013])
    np.testing.assert_allclose(cb.eff(e),
                               [0.0338679, 0.0152847, 0.00714092, 0.00403744], rtol=1E-4)
    assert (cb._engcal_model, cb._rescal_model, cb._effcal_model) == ('quadratic', 'linear', 'vidmar-5')
    assert cb.diagnostics.set_index('fit')['model'].to_dict() == {
        'engcal': 'quadratic', 'rescal': 'linear', 'effcal': 'vidmar-5'}


@requires_data('decay')
def test_loglog_model_and_json_roundtrip(tmp_path, capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='loglog')
    out = capsys.readouterr().out
    assert 'effcal model: loglog-4 (user-selected)' in out
    assert 'effcal [loglog-4] fit to' in out
    assert cb._effcal_model == 'loglog-4'
    assert len(cb.effcal) == 5  # order 4 -> 5 coefficients, same length as vidmar-5
    assert cb.diagnostics.set_index('fit').loc['effcal', 'model'] == 'loglog-4'
    # the tag is load-bearing: the same parameters dispatched as vidmar give
    # a different (wrong) efficiency
    e = np.array([300.0, 800.0])
    ll = cb.eff(e)
    np.testing.assert_allclose(ll, np.exp(np.polyval(cb.effcal[::-1], np.log(e))), rtol=1E-12)
    assert not np.allclose(ll, ci.Calibration().eff(e, cb.effcal))
    # the tidy table's residuals must use the loglog model too (same length
    # collision as above - a vidmar dispatch produces garbage residuals)
    d = cb.effcal_data
    expect = d['efficiency'].to_numpy() - np.exp(np.polyval(cb.effcal[::-1], np.log(d['energy'].to_numpy())))
    np.testing.assert_allclose(d['residual'].to_numpy(), expect, rtol=1E-9)
    # round trip preserves tag, range and values
    f = str(tmp_path / 'll.json')
    cb.saveas(f)
    cb2 = ci.Calibration(f)
    assert cb2._effcal_model == 'loglog-4'
    assert cb2._effcal_erange == cb._effcal_erange
    np.testing.assert_allclose(cb2.eff(e), ll, rtol=1E-12)
    js = json.load(open(f))
    assert js['effcal_model'] == 'loglog-4'
    assert len(js['effcal_erange']) == 2


def test_loglog_unc_eff_exact_gradient():
    # the loglog band comes from the exact gradient d(eff)/d(a_i) = eff*ln(E)^i:
    # with a diagonal covariance it is eff*sqrt(sum_i V_ii ln(E)^2i) exactly
    # (the generic finite-difference path is orders of magnitude off for
    # coefficients that live in the exponent)
    cb = ci.Calibration()
    c = np.array([-9.0, 1.2, -0.15])
    V = np.diag([1E-4, 4E-6, 1E-7])
    e = np.array([150.0, 500.0, 1300.0])
    got = cb.unc_eff(e, c, V, model='loglog')
    eff = np.exp(np.polyval(c[::-1], np.log(e)))
    lnE = np.log(e)
    expect = eff*np.sqrt(sum(V[i][i]*lnE**(2*i) for i in range(3)))
    np.testing.assert_allclose(got, expect, rtol=1E-12)


@requires_data('decay')
def test_untagged_legacy_json(tmp_path):
    # a file without model tags (older curie) loads with infer-by-length
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES)
    f = str(tmp_path / 'cb.json')
    cb.saveas(f)
    js = json.load(open(f))
    for key in ['engcal_model', 'rescal_model', 'effcal_model', 'effcal_erange']:
        js.pop(key, None)
    old = str(tmp_path / 'cb_old.json')
    json.dump(js, open(old, 'w'))
    cb2 = ci.Calibration(old)
    assert cb2._effcal_model is None and cb2._effcal_erange is None
    e = np.array([300.0, 800.0])
    np.testing.assert_allclose(cb2.eff(e), cb.eff(e), rtol=1E-12)
    np.testing.assert_allclose(cb2.eng([1000, 5000]), cb.eng([1000, 5000]), rtol=1E-12)


@requires_data('decay')
def test_cubic_engcal_and_inverse():
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, engcal_model='cubic')
    assert cb._engcal_model == 'cubic' and len(cb.engcal) == 4
    # numeric inverse round-trips to within one channel width across the range
    # (including high energies, where an integer channel overflows if cubed
    # as int32)
    for E in [150.0, 500.0, 1000.0, 1408.0, 1500.0]:
        ch = cb.map_channel(E)
        assert abs(float(cb.eng(ch))-E) < float(cb.eng(int(ch)+1)-cb.eng(int(ch)))


def test_cubic_non_monotonic_warning(capsys, monkeypatch):
    real = cal_mod.curve_fit
    state = {'n': 0}
    def rigged(fn, x, y, **kw):
        state['n'] += 1
        if state['n'] == 1:
            # a cubic that turns over inside the fitted span
            return np.array([0.0, 1.0, 0.0, -1E-4]), np.eye(4)
        return real(fn, x, y, **kw)
    monkeypatch.setattr(cal_mod, 'curve_fit', rigged)
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, engcal_model='cubic')
    out = capsys.readouterr().out
    assert 'cubic energy calibration is non-monotonic' in out


@requires_data('decay')
def test_sqrt_quad_rescal():
    sp = eu_sp()
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES, rescal_model='sqrt_quad')
    assert cb._rescal_model == 'sqrt_quad' and len(cb.rescal) == 3
    assert cb.diagnostics.set_index('fit').loc['rescal', 'model'] == 'sqrt_quad'
    # agrees with the linear form to well within the width scatter over the
    # fitted range
    cb_lin = ci.Calibration()
    cb_lin.calibrate([eu_sp()], sources=EU_SOURCES)
    ch = np.linspace(1000, 8000, 15)
    assert np.all(np.abs(cb.res(ch)-cb_lin.res(ch))/cb_lin.res(ch) < 0.25)


@requires_data('decay')
def test_eff_points_augment(capsys):
    ep = pd.DataFrame({'energy': [2000.0, 2500.0], 'efficiency': [0.003, 0.0025],
                       'unc_efficiency': [0.0003, 0.00025], 'isotope': ['56CO', '56CO']})
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=ep)
    out = capsys.readouterr().out
    assert 'effcal includes 2 user-supplied points (eff_points)' in out
    assert '+2 user-supplied points' in out
    assert cb._effcal_erange[1] == 2500.0
    d = cb.effcal_data
    assert (d['isotope'] == '56CO').sum() == 2
    assert int(cb.diagnostics.set_index('fit').loc['effcal', 'n_points']) == 44


@requires_data('decay')
def test_threshold_knobs(capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_max_error=0.1, outlier_sigma=1E9)
    out = capsys.readouterr().out
    assert 'dropped: unc>10% of value' in out
    assert 'outliers' not in out.split('effcal')[-1]
    assert (cb.effcal_data.loc[~cb.effcal_data['used'], 'reason'] == 'unc>10%').all()


@requires_data('decay')
def test_extrapolation_warning_once(capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES)
    capsys.readouterr()
    _ = cb.eff(3000.0)
    _ = cb.eff(3200.0)
    out = capsys.readouterr().out
    assert out.count('outside the fitted range') == 1
    assert '[WARNING]' in out


@requires_data('decay')
def test_forced_vidmar_7(capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='vidmar-7')
    out = capsys.readouterr().out
    assert 'effcal model: vidmar-7 (user-selected)' in out
    assert cb._effcal_model == 'vidmar-7' and len(cb.effcal) == 7
