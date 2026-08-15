"""Tests for the selectable calibration models and Calibration.fit_config.

Covers the model tags (explicit state, .json persistence, infer-by-length
backcompat), the loglog efficiency model (whose parameter length can collide
with Vidmar's, making the tag load-bearing), the cubic energy calibration and
its numeric inverse, the sqrt_quad resolution model, the eff_points
augment-and-refit, the threshold knobs, the extrapolation warning and its
per-direction warn-once behavior, the physical range of the fitted Vidmar
parameters (peak-to-total at most one), the low-energy route to the
7-parameter form, the low-energy disagreement announcement, and the fitted
range stored from the points the fit kept.
"""
import json
import re

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
    # curve values re-recorded 2026-08-15 for the physical efficiency bounds
    # (kappa <= 1, effective chord length <= 10 cm, l0 boxed to 0.5-3 cm),
    # which reshape the fitted vidmar-5 curve by up to ~0.8% here - largest at
    # the high-energy tail where the tighter length and kappa bounds bite.
    # The model stays vidmar-5: this spectrum has one efficiency point below
    # 150 keV (121.8 keV), one short of the two the low-energy route needs.
    # Recorded against the v2 data generation, as the previous values were
    # (originally recorded at the doublet-merge fix on fitting-0.2.0) - a
    # change means a default moved.  The efficiency CURVE is pinned rather than
    # the raw parameters: the effective-length parameter is nearly degenerate
    # once saturated, so different scipy/BLAS stacks converge to different
    # parameterizations of the same curve (observed cross-platform curve
    # reproducibility ~2E-6; tolerance gives 50x margin)
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES)
    e = np.array([121.7817, 344.2785, 778.9045, 1408.013])
    np.testing.assert_allclose(cb.eff(e),
                               [0.0339014, 0.0152819, 0.0071402, 0.0039618], rtol=1E-4)
    assert (cb._engcal_model, cb._rescal_model, cb._effcal_model) == ('quadratic', 'linear', 'vidmar-5')
    assert cb.diagnostics.set_index('fit')['model'].to_dict() == {
        'engcal': 'quadratic', 'rescal': 'linear', 'effcal': 'vidmar-5'}
    # no point below 150 keV is discarded here, so the disagreement flag is absent
    assert 'low_e_disagreement' not in cb.diagnostics.set_index('fit').loc['effcal', 'flags']


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
    # each direction warns once: an above-range warning must not swallow the
    # below-range caution a later call deserves, since below the lowest fitted
    # point the Vidmar shape is an unconstrained extrapolation
    cb._extrap_warned = cb._extrap_low_warned = False
    capsys.readouterr()
    _ = cb.eff(3000.0)
    _ = cb.eff(50.0)
    out = capsys.readouterr().out
    assert out.count('outside the fitted range') == 2
    # the below-range caution is direction-neutral (vidmar-7 rolls over and
    # runs low below its peak, so "runs high" would mislead the low-energy case)
    assert 'unconstrained extrapolation' in out
    # and the below-range caution rides on the below direction only: a bare
    # above-range evaluation does not carry it
    cb._extrap_warned = cb._extrap_low_warned = False
    capsys.readouterr()
    _ = cb.eff(3000.0)
    out = capsys.readouterr().out
    assert 'outside the fitted range' in out and 'unconstrained extrapolation' not in out


@requires_data('decay')
def test_low_energy_disagreement_announced(capsys):
    # a user-supplied point at 100 keV, far above anything the bounded forms
    # can reach: the exclusion stands - one point cannot outvote the fit - but
    # it is announced with its signed residual, flagged on the diagnostics row,
    # and stays readable in effcal_data.  Below 150 keV a clipped point is the
    # only evidence of a disagreement a user would ever see, so clipping it
    # quietly is what would hide a low-energy regression
    cb = ci.Calibration()
    pts = pd.DataFrame([{'energy': 100.0, 'efficiency': 0.06, 'unc_efficiency': 0.005}])
    capsys.readouterr()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=pts)
    out = capsys.readouterr().out
    assert 'disagrees with 1 point(s) below 150 keV' in out and '[WARNING]' in out
    assert '100.0 keV +' in out  # signed residual, above the curve
    row = cb.diagnostics.set_index('fit').loc['effcal']
    assert 'low_e_disagreement' in row['flags']
    assert 'disagreement below 150 keV: 1 point(s) excluded' in row['message']
    dropped = cb.effcal_data
    dropped = dropped[~dropped['used'] & (dropped['energy'] == 100.0)]
    assert len(dropped) == 1 and dropped['reason'].iloc[0].startswith('outlier')


@requires_data('decay')
def test_effcal_erange_from_kept_points(capsys):
    # a low-energy point the outlier screen removed must not extend the fitted
    # range: the stored floor is the lowest KEPT point, so the gap the
    # exclusion opened is guarded by the extrapolation warning rather than
    # silently reported as in-range
    cb = ci.Calibration()
    pts = pd.DataFrame([{'energy': 100.0, 'efficiency': 0.06, 'unc_efficiency': 0.005}])
    cb.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=pts)
    # the 100 keV point was clipped; the floor is the lowest surviving point
    used_lo = cb.effcal_data.loc[cb.effcal_data['used'], 'energy'].min()
    assert cb._effcal_erange[0] == pytest.approx(used_lo)
    assert cb._effcal_erange[0] > 100.0
    # evaluating in the gap the exclusion opened now warns
    capsys.readouterr()
    _ = cb.eff(110.0)
    out = capsys.readouterr().out
    assert 'outside the fitted range {0:.1f}'.format(cb._effcal_erange[0]) in out


def _peak_to_total(cb, energy):
    # the peak-to-total ratio implied by the fitted parameters: photoelectric
    # absorption plus the recovered-Compton fraction, over the total attenuation
    sa, l, alpha, l0, kappa = tuple(np.asarray(cb.effcal, dtype=np.float64)[:5])
    kappa_eff = kappa*(1.0-np.exp(-(cal_mod._MU(energy)*l0)**alpha))
    return (cal_mod._TAU(energy)+kappa_eff*cal_mod._SIGMA(energy))/cal_mod._MU(energy)


def _kappa_driving_points():
    # efficiency points sampled from a Vidmar-5 curve with kappa = 2.5 - an
    # unphysical peak-to-total above one - at tight uncertainty, so the fit is
    # pulled toward that kappa.  The unbounded fit follows it (kappa runs to
    # its old ceiling, P/T well above one); the bounded fit must cap kappa at
    # one and hold P/T at or below one
    synth = np.array([0.03, 5.0, 1.6, 1.2, 2.5])
    e = np.linspace(120.0, 2400.0, 40)
    eff = ci.Calibration().eff(e, synth)
    return pd.DataFrame({'energy': e, 'efficiency': eff,
                         'unc_efficiency': eff*0.002, 'isotope': ['synthetic']*len(e)})


@requires_data('decay')
def test_effcal_physical_range():
    # one registered photon can make at most one full-energy count, so the
    # fitted peak-to-total ratio cannot exceed one.  It follows from kappa <= 1;
    # a fit that escapes the bound reports a detector that counts more
    # full-energy events than the interactions it recorded.  Both the natural
    # example fits and a fit deliberately pulled toward an unphysical kappa
    # must stay at or below one
    driving = _kappa_driving_points()
    cases = [('auto', {}, None), ('vidmar-5', {'effcal_model': 'vidmar-5'}, None),
             ('vidmar-7', {'effcal_model': 'vidmar-7'}, None),
             ('vidmar-5 pulled', {'effcal_model': 'vidmar-5'}, driving),
             ('vidmar-7 pulled', {'effcal_model': 'vidmar-7'}, driving)]
    for name, kw, ep in cases:
        cb = ci.Calibration()
        cb.calibrate([eu_sp()], sources=EU_SOURCES,
                     **kw, **({} if ep is None else {'eff_points': ep}))
        lo, hi = cb._effcal_erange
        e = np.linspace(lo, hi, 500)
        p_over_t = _peak_to_total(cb, e)
        assert np.all(p_over_t <= 1.0), '{0}: max P/T = {1:.4f}'.format(name, p_over_t.max())
        assert cb.effcal[4] <= 1.0 + 1E-9, '{0}: kappa = {1:.4f}'.format(name, cb.effcal[4])


@requires_data('decay')
def test_vidmar7_low_energy_routing(capsys):
    # below 150 keV the entrance window and dead layer shape the curve, so the
    # 7-parameter form is tried once two fit points land there.  The example
    # spectrum supplies one (121.8 keV); a user-supplied point at 59.5 keV is
    # the second, and with no x-ray peaks in the fit those two are the whole
    # trigger.  Routing means both forms are fit and compared; which one is
    # kept is the model-selection decision tested separately
    sp = eu_sp()
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES,
                 eff_points=pd.DataFrame({'energy': [59.5], 'efficiency': [0.0385],
                                          'unc_efficiency': [0.0019], 'isotope': ['241AM']}))
    out = capsys.readouterr().out
    assert not sp.fit_config['xrays']
    # the trigger counts the points that survived the pre-fit uncertainty cut,
    # not the points effcal_data adds back
    in_fit = cb.effcal_data[~cb.effcal_data['reason'].str.startswith('unc>')]
    assert int(np.sum(in_fit['energy'].to_numpy() < cal_mod._VIDMAR_LOW_E)) >= 2
    # both forms were fit and compared: the selection message names each one
    sel = [l for l in out.splitlines() if 'effcal model selection' in l]
    assert len(sel) == 1 and 'vidmar-5' in sel[0] and 'vidmar-7' in sel[0]
    # a single point below the threshold does not widen the trigger: without
    # the user point the same spectrum takes the 5-parameter route directly
    cb2 = ci.Calibration()
    cb2.calibrate([eu_sp()], sources=EU_SOURCES)
    out2 = capsys.readouterr().out
    assert 'vidmar-5 (auto - no x-ray peaks fit, fewer than two points below 150 keV)' in out2
    assert cb2._effcal_model == 'vidmar-5' and len(cb2.effcal) == 5


@requires_data('decay')
def test_effcal_selection_prefers_smaller_inflation(capsys):
    # the 5-vs-7 choice is decided on the uncertainty inflation each form needed
    # to reach a self-consistent fit (smaller is better), not on the converged
    # whitened chi2 the inflation loop drives to one by construction.  On this
    # spectrum the two rules disagree: the 7-parameter form reaches a marginally
    # smaller converged chi2 (which the chi2 rule would keep) but a larger
    # inflation, so the inflation rule keeps the simpler 5-parameter form
    sp = eu_sp()
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES,
                 eff_points=pd.DataFrame({'energy': [59.5], 'efficiency': [0.0385],
                                          'unc_efficiency': [0.0019], 'isotope': ['241AM']}))
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if 'effcal model selection' in l][0]
    # the kept form is the one with the smaller uncertainty inflation
    assert 'smaller uncertainty inflation' in line
    assert cb._effcal_model == 'vidmar-5' and len(cb.effcal) == 5
    # the message reports the kept form first: its S2 is the smaller of the two
    s2 = [float(v) for v in re.findall(r'S2=([0-9.eE+-]+)', line)]
    assert len(s2) == 2 and s2[0] <= s2[1]


@requires_data('decay')
def test_forced_vidmar_7(capsys):
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='vidmar-7')
    out = capsys.readouterr().out
    assert 'effcal model: vidmar-7 (user-selected)' in out
    assert cb._effcal_model == 'vidmar-7' and len(cb.effcal) == 7


@requires_data('decay')
def test_effcal_underdetermined_vidmar_guard():
    # a Vidmar fit cannot be determined by fewer points than it has free
    # parameters.  A tight uncertainty cut drops the poorly-quantified measured
    # points, leaving one; two sub-150 keV user points route to the low-energy
    # (7-parameter) branch with only three fit points.  Without a guard that
    # ships a 3-point 7-parameter fit - a rank-deficient covariance and an
    # unphysical curve that rises to a peak near 400 keV.  The 5-parameter form
    # is the floor: mirror the loglog guard and raise, matching its wording
    ep = pd.DataFrame([
        {'isotope': 'user', 'energy': 50.0, 'efficiency': 0.0060, 'unc_efficiency': 0.0003},
        {'isotope': 'user', 'energy': 90.0, 'efficiency': 0.0075, 'unc_efficiency': 0.0004}])
    cb = ci.Calibration()
    with pytest.raises(ValueError, match=r'effcal_model vidmar-5 needs at least 5 efficiency points to fit \(have 3\)'):
        cb.calibrate([eu_sp()], sources=EU_SOURCES,
                     effcal_max_error=0.008, outlier_sigma=1E9, eff_points=ep)
