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
    # re-recorded 2026-08-16 on the combined tree: the constrained
    # close-doublet fit recovers the 152Eu 1084.0 keV line (previously 138%
    # rel. unc. from the unconstrained width degeneracy), adding it to the
    # bounded efficiency fit (42 to 43 base points); measured together with
    # the physical-bounds values above, largest move +1.0% at 1408 keV.
    # The recovered area still carries an intrinsic low-SNR positive bias of
    # order +10% at this line's ~41:1 intensity ratio (ensemble-measured at
    # full SNR-screen survival), inherent to weak-line fitting rather than
    # introduced by the constraint.
    np.testing.assert_allclose(cb.eff(e),
                               [0.03390682, 0.01527646, 0.00716950, 0.00400189], rtol=1E-4)
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
    # 45 = 43 base 152Eu points (the constrained close-doublet fit recovers
    # the 1084.0 keV line) + 2 user-supplied points. The recovered area still
    # carries an intrinsic low-SNR positive bias of order +10% at this line's
    # ~41:1 intensity ratio (ensemble-measured at full SNR-screen survival),
    # inherent to weak-line fitting rather than introduced by the constraint.
    assert int(cb.diagnostics.set_index('fit').loc['effcal', 'n_points']) == 45


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


def _starve_unlimited_eff_fits(monkeypatch):
    # an efficiency-fit call (the only calls that pass bounds) that names no
    # evaluation limit gets a limit of a few evaluations, far below what any
    # fit needs; every efficiency call is recorded with the limit it named
    real = cal_mod.curve_fit
    limits = []
    def starved(fn, x, y, **kw):
        if 'bounds' in kw:
            limits.append(kw.get('maxfev'))
            if kw.get('maxfev') is None:
                kw = dict(kw, maxfev=3)
        return real(fn, x, y, **kw)
    monkeypatch.setattr(cal_mod, 'curve_fit', starved)
    return limits


@requires_data('decay')
@pytest.mark.parametrize('model', ['vidmar-5', 'loglog'])
def test_effcal_fit_names_evaluation_limit(model, monkeypatch):
    # the efficiency fit passes its own evaluation limit on every call - the
    # first, independent-uncertainty call and the correlated-uncertainty calls
    # after it - on the bounded (vidmar) path and on the unbounded loglog path,
    # so it converges even where the optimizer default would stop it
    limits = _starve_unlimited_eff_fits(monkeypatch)
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_model=model)
    assert len(limits) >= 2
    assert limits == [cal_mod._EFF_MAXFEV]*len(limits)
    assert np.all(np.isfinite(cb.effcal))


def _rig_curve_fit(monkeypatch, select, action):
    # calls for which select(kw) holds get action(kw) instead of a fit; the
    # efficiency fit is the only caller that passes bounds, its first pass the
    # only one with 1-D sigma, and p0 has one entry per parameter of the form
    real = cal_mod.curve_fit
    def rigged(fn, x, y, **kw):
        if select(kw):
            return action(kw)
        return real(fn, x, y, **kw)
    monkeypatch.setattr(cal_mod, 'curve_fit', rigged)


def _raise(err):
    def action(kw):
        raise err
    return action


EFF_PASSES = {1: 'independent', 2: 'correlated'}


@requires_data('decay')
@pytest.mark.parametrize('model,tag', [('vidmar-5', 'vidmar-5'), ('loglog', 'loglog-4')])
@pytest.mark.parametrize('ndim', [1, 2])
def test_effcal_nonconvergence_names_model_and_pass(model, tag, ndim, monkeypatch):
    # a pass of the efficiency fit that does not converge stops the calibration
    # with a curie RuntimeError raised from the optimizer's error, naming the
    # model, the pass and the number of fit points
    forced = RuntimeError('Optimal parameters not found: forced')
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and np.ndim(kw['sigma']) == ndim, _raise(forced))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model=model)
    msg = str(ei.value)
    assert msg.startswith('Calibration.calibrate:')
    assert 'the {0} efficiency fit did not converge in its pass with {1} uncertainties'.format(tag, EFF_PASSES[ndim]) in msg
    assert re.search(r'\(\d+ fit points; Optimal parameters not found: forced\)', msg)
    assert 'effcal_model' in msg and 'peak fits' in msg
    assert ei.value.__cause__ is forced


@requires_data('decay')
def test_effcal_nonfinite_model_values_raise(monkeypatch):
    # the unbounded loglog search can step to parameters where the model
    # overflows; the optimizer then stops with a ValueError on the non-finite
    # residuals, which is a fit that did not converge
    forced = ValueError('array must not contain infs or NaNs')
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and np.ndim(kw['sigma']) == 2, _raise(forced))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='loglog')
    msg = str(ei.value)
    assert msg.startswith('Calibration.calibrate: the loglog-4 efficiency fit did not converge (non-finite model values) in its pass with correlated uncertainties')
    assert ei.value.__cause__ is forced


@requires_data('decay')
@pytest.mark.parametrize('ndim', [1, 2])
def test_effcal_nonfinite_parameters_raise(ndim, monkeypatch):
    # a pass that returns non-finite parameters is checked before they are used
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and np.ndim(kw['sigma']) == ndim,
                   lambda kw: (np.full(len(kw['p0']), np.nan), np.eye(len(kw['p0']))))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='vidmar-5')
    assert str(ei.value).startswith('Calibration.calibrate: the vidmar-5 efficiency fit returned non-finite parameters in its pass with {0} uncertainties'.format(EFF_PASSES[ndim]))


USER_59 = pd.DataFrame({'energy': [59.5], 'efficiency': [0.0385],
                        'unc_efficiency': [0.0019], 'isotope': ['241AM']})


@requires_data('decay')
def test_effcal_auto_five_parameter_failure_raises(capsys, monkeypatch):
    # on the low-energy route (setup of test_vidmar7_low_energy_routing) the
    # 5-parameter form is fit once, outside the 7-parameter fallback: its
    # failure raises, and is not reported as a 7-parameter failure
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and len(kw['p0']) == 5,
                   _raise(RuntimeError('Optimal parameters not found: forced')))
    with pytest.raises(RuntimeError, match=r'^Calibration\.calibrate: the vidmar-5 efficiency fit did not converge'):
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, eff_points=USER_59)
    assert '7-parameter fit failed' not in capsys.readouterr().out


@requires_data('decay')
def test_effcal_auto_seven_parameter_failure_falls_back(capsys, monkeypatch):
    # a failed 7-parameter fit warns and keeps the 5-parameter result already
    # in hand, with its chi-square before inflation, without fitting it again
    ref = ci.Calibration()
    ref.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=USER_59, effcal_model='vidmar-5')
    real = cal_mod.curve_fit
    first5 = []
    def rigged(fn, x, y, **kw):
        if 'bounds' in kw and len(kw['p0']) == 7:
            raise RuntimeError('Optimal parameters not found: forced')
        if 'bounds' in kw and len(kw['p0']) == 5 and np.ndim(kw['sigma']) == 1:
            first5.append(1)
        return real(fn, x, y, **kw)
    monkeypatch.setattr(cal_mod, 'curve_fit', rigged)
    capsys.readouterr()
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=USER_59)
    out = capsys.readouterr().out
    assert 'effcal 7-parameter fit failed (Calibration.calibrate: the vidmar-7 efficiency fit did not converge' in out
    assert 'using the 5-parameter form' in out
    assert len(first5) == 1
    assert cb._effcal_model == 'vidmar-5'
    np.testing.assert_array_equal(cb.effcal, ref.effcal)
    np.testing.assert_array_equal(cb.unc_effcal, ref.unc_effcal)
    got, want = cb.diagnostics.set_index('fit').loc['effcal'], ref.diagnostics.set_index('fit').loc['effcal']
    assert (got['chi2'], got['scale_factor']) == (want['chi2'], want['scale_factor'])


@requires_data('decay')
@pytest.mark.parametrize('fit,n_par', [('energy', 3), ('resolution', 2)])
def test_engcal_rescal_nonconvergence_raise(fit, n_par, monkeypatch):
    # the energy (quadratic, 3 parameters) and resolution (linear, 2) fits of
    # the example spectrum pass no bounds; a forced non-convergence of either
    # becomes a curie RuntimeError naming the fit and its model
    forced = RuntimeError('Optimal parameters not found: forced')
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' not in kw and len(kw['p0']) == n_par, _raise(forced))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES)
    model = {'energy': 'engcal_model quadratic', 'resolution': 'rescal_model linear'}[fit]
    assert str(ei.value).startswith('Calibration.calibrate: the {0} calibration fit ({1}) did not converge'.format(fit, model))
    assert ei.value.__cause__ is forced


def _cut_between(rel, n_keep):
    # a relative-uncertainty cut that keeps exactly the n_keep best points
    r = np.sort(np.asarray(rel, dtype=np.float64))
    assert r[n_keep-1] < r[n_keep]
    return 0.5*(r[n_keep-1]+r[n_keep])


@requires_data('decay')
@pytest.mark.parametrize('fit', ['engcal', 'rescal'])
def test_engcal_rescal_too_few_points(fit):
    # an energy or resolution fit with fewer surviving points than parameters
    # stops with a ValueError naming the model, the points it needs and the
    # points that survived the uncertainty filter (the cut is set from the
    # relative uncertainties of an unfiltered calibration of the same spectrum)
    ref = ci.Calibration()
    ref.calibrate([eu_sp()], sources=EU_SOURCES)
    if fit == 'engcal':
        d, keep = ref.engcal_data, 2
        cut = _cut_between(d['unc_channel']/d['energy'], keep)
        head = 'engcal_model quadratic needs at least 3 energy calibration points to fit (have 2)'
    else:
        d, keep = ref.rescal_data, 1
        cut = _cut_between(d['unc_width']/d['width'], keep)
        head = 'rescal_model linear needs at least 2 resolution calibration points to fit (have 1)'
    with pytest.raises(ValueError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, **{fit+'_max_error': cut})
    msg = str(ei.value)
    assert msg.startswith('Calibration.calibrate: '+head)
    assert 'dropped {0} of {1} points'.format(len(d)-keep, len(d)) in msg


@requires_data('decay')
@pytest.mark.parametrize('model,need', [('vidmar-5', 5), ('loglog', 5)])
def test_effcal_too_few_points_message(model, need):
    # the too-few-efficiency-points messages keep their wording through
    # "(have N)", then state how many measured points the uncertainty filter
    # dropped and name the energy calibration as a cause to check.  Setup of
    # test_effcal_underdetermined_vidmar_guard: one measured point survives the
    # cut, two user points join it
    ep = pd.DataFrame([
        {'isotope': 'user', 'energy': 50.0, 'efficiency': 0.0060, 'unc_efficiency': 0.0003},
        {'isotope': 'user', 'energy': 90.0, 'efficiency': 0.0075, 'unc_efficiency': 0.0004}])
    with pytest.raises(ValueError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model=model,
                                   effcal_max_error=0.008, outlier_sigma=1E9, eff_points=ep)
    msg = str(ei.value)
    tag = {'vidmar-5': 'vidmar-5', 'loglog': 'loglog-4'}[model]
    head = 'Calibration.calibrate: effcal_model {0} needs at least {1} efficiency points to fit (have 3)'.format(tag, need)
    assert msg.startswith(head)
    m = re.match(re.escape(head)+r'; the uncertainty filter \(unc>0\.8% of value\) dropped (\d+) of (\d+) measured points', msg)
    assert m and int(m.group(2))-int(m.group(1)) == 1
    assert 'energy calibration' in msg[len(head):] and 'peak fits' in msg[len(head):]
