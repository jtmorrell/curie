"""Tests for the selectable calibration models and Calibration.fit_config.

Covers the model tags (explicit state, .json persistence, infer-by-length
backcompat), the loglog efficiency model (whose parameter length can collide
with Vidmar's, making the tag load-bearing), the cubic energy calibration and
its numeric inverse, the sqrt_quad resolution model, the eff_points
augment-and-refit, the threshold knobs, the extrapolation warning and its
per-direction warn-once behavior, the physical range of the fitted Vidmar
parameters (peak-to-total at most one), the low-energy route to the
7-parameter form, the low-energy disagreement announcement, the fitted
range stored from the points the fit kept, and the line-intensity floor on
the measured efficiency points (effcal_I_min).
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

import curie as ci
import curie.calibration as cal_mod
from conftest import EXAMPLES_DIR, requires_data, requires_pairs

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
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=0, summing=False)
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
    cb.calibrate([eu_sp()], sources=EU_SOURCES, eff_points=ep, effcal_I_min=0)
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
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_max_error=0.1, outlier_sigma=1E9, effcal_I_min=0)
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
                                          'unc_efficiency': [0.0019], 'isotope': ['241AM']}),
                 summing=False)
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
    # residuals, which becomes the curie RuntimeError with the optimizer's text
    forced = ValueError('array must not contain infs or NaNs')
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and np.ndim(kw['sigma']) == 2, _raise(forced))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='loglog')
    msg = str(ei.value)
    assert msg.startswith('Calibration.calibrate: the loglog-4 efficiency fit stopped with an optimizer error in its pass with correlated uncertainties')
    assert re.search(r'\(\d+ fit points; array must not contain infs or NaNs\)', msg)
    assert ei.value.__cause__ is forced


@requires_data('decay')
def test_effcal_optimizer_value_error_not_called_nonfinite(monkeypatch):
    # the optimizer also raises ValueError where no model value is non-finite:
    # a start outside the bounds (a user efficiency point with zero uncertainty
    # gives a NaN solid-angle start) or a covariance that is not positive
    # definite.  The message does not attribute these to non-finite model values
    forced = ValueError('Initial guess is outside of provided bounds')
    _rig_curve_fit(monkeypatch, lambda kw: 'bounds' in kw and np.ndim(kw['sigma']) == 1, _raise(forced))
    with pytest.raises(RuntimeError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_model='vidmar-5')
    msg = str(ei.value)
    assert msg.startswith('Calibration.calibrate: the vidmar-5 efficiency fit')
    assert 'non-finite model values' not in msg
    assert 'Initial guess is outside of provided bounds' in msg
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
    # the warning carries the reason without a second "Calibration.calibrate:"
    # prefix or the stop-the-calibration advice, since the calibration goes on
    assert 'Calibration.calibrate: effcal 7-parameter fit failed (the vidmar-7 efficiency fit did not converge in its pass with' in out
    assert 'failed (Calibration.calibrate:' not in out and 'try another effcal_model' not in out
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
                                   effcal_max_error=0.008, outlier_sigma=1E9, eff_points=ep, effcal_I_min=0)
    msg = str(ei.value)
    tag = {'vidmar-5': 'vidmar-5', 'loglog': 'loglog-4'}[model]
    head = 'Calibration.calibrate: effcal_model {0} needs at least {1} efficiency points to fit (have 3)'.format(tag, need)
    assert msg.startswith(head)
    m = re.match(re.escape(head)+r'; the uncertainty filter \(unc>0\.8% of value\) dropped (\d+) of (\d+) measured points', msg)
    assert m and int(m.group(2))-int(m.group(1)) == 1
    assert 'energy calibration' in msg[len(head):] and 'peak fits' in msg[len(head):]
    # vidmar-5 is the floor of the Vidmar forms: no lower order to advise
    assert ('lower the order' in msg) == (model == 'loglog')


# the effcal [vidmar-5] summary line of the example spectrum before the
# line-intensity floor existed, recorded 2026-11-10 at commit 617dcd7 from
# ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES); effcal_I_min=0
# must reproduce it
PRE_FLOOR_EFFCAL_INFO = ('[INFO] Calibration.calibrate: effcal [vidmar-5] fit to 43/44 points '
                         '(1 dropped: unc>33% of value); 6 outliers: residual >3.16 sigma '
                         '[152EU 244.7, 152EU 444.0, 152EU 867.4, 152EU 1005.3, 152EU 1109.2, '
                         '152EU 1528.1 keV]; chi2/dof=24; independent uncertainty components scaled x5.41')


def _effcal_info(out):
    return [l for l in out.splitlines() if 'Calibration.calibrate: effcal [' in l]


def _line_intensities(sp):
    # per-decay intensity of each source line in the peak table, keyed by
    # energy as the calibration points store it; read before calibrate(),
    # which clears the peak table of its spectra
    pk = sp.peaks
    pk = pk[pk['isotope'] == '152EU']
    return dict(zip(pk['energy'].round(4), pk['intensity']))


@requires_data('decay')
def test_effcal_I_min_default_floor(capsys):
    # the default floor of 1% leaves out every point whose line intensity is
    # below 1%: none is in the fit, each is in effcal_data as not used with the
    # reason 'intensity<1%', and the summary line counts them in a clause of
    # their own
    sp = eu_sp()
    inten = _line_intensities(sp)
    n_low = sum(100.0*i < 1.0 for i in inten.values())
    capsys.readouterr()
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES, summing=False)
    out = capsys.readouterr().out
    used = cb._calib_data['effcal']['energy']
    assert len(used) and all(100.0*inten[round(e, 4)] >= 1.0 for e in used)
    d = cb.effcal_data
    low = d[[100.0*inten[round(e, 4)] < 1.0 for e in d['energy']]]
    assert len(low) == n_low > 0
    assert (~low['used']).all() and (low['reason'] == 'intensity<1%').all()
    assert (d['reason'] == 'intensity<1%').sum() == n_low
    info = _effcal_info(out)
    assert len(info) == 1
    assert '; {0} dropped: line intensity <1% (effcal_I_min)'.format(n_low) in info[0]
    row = cb.diagnostics.set_index('fit').loc['effcal']
    assert row['message'] in info[0]
    assert int(row['n_points']) == len(d) - int((~d['used']).sum()) + int((d['reason'].str.startswith('outlier')).sum())
    # the default curve, recorded 2026-11-10 from this test's calibrate() call
    # with the floor applied to commit 617dcd7; the floor
    # leaves 14 of the 44 152Eu points in the fit and moves the curve of
    # test_zero_config_identity by -0.04% at 121.8 keV to -2.3% at 1408 keV.
    # Tolerance as in test_zero_config_identity
    e = np.array([121.7817, 344.2785, 778.9045, 1408.013])
    np.testing.assert_allclose(cb.eff(e), [0.03389453, 0.01528831, 0.00712934, 0.00390890], rtol=1E-4)
    assert int(row['n_points']) == 14 and row['model'] == 'vidmar-5'


@requires_data('decay')
def test_effcal_I_min_zero_is_pre_floor_selection(capsys):
    # effcal_I_min=0 keeps every line: the point count, the dropped points and
    # their reasons, and the summary line are those recorded before the floor
    # existed (the curve itself is pinned in test_zero_config_identity)
    capsys.readouterr()
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=0, summing=False)
    out = capsys.readouterr().out
    assert _effcal_info(out) == [PRE_FLOOR_EFFCAL_INFO]
    row = cb.diagnostics.set_index('fit').loc['effcal']
    assert (int(row['n_points']), int(row['n_dropped'])) == (43, 7)
    reasons = cb.effcal_data.loc[~cb.effcal_data['used'], 'reason'].tolist()
    assert reasons == ['unc>33%'] + ['outlier >3.16 sigma']*6
    assert 'intensity<' not in out.split('effcal [')[-1]


@requires_data('decay')
@requires_pairs()
def test_summing_default_curve():
    # the default curve, with the coincidence-summing correction, recorded
    # 2026-11-10 from this calibrate() call when the correction became the
    # default.  It lies above the summing=False curve of
    # test_effcal_I_min_default_floor by 5.9% at 121.8 keV, 4.0% at 344.3
    # keV, 3.2% at 778.9 keV and 5.1% at 1408 keV: at 7 cm the summing-out
    # loss of each line exceeds its gain, so the measured points lie below the
    # curve that is free of summing.  Tolerance as in test_zero_config_identity
    if not cal_mod._pair_table_usable():
        pytest.skip('the installed decay database fails the data-generation check')
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES)
    off = ci.Calibration()
    off.calibrate([eu_sp()], sources=EU_SOURCES, summing=False)
    e = np.array([121.7817, 344.2785, 778.9045, 1408.013])
    np.testing.assert_allclose(cb.eff(e), [0.03588573, 0.01590491, 0.00735884, 0.00410683], rtol=1E-4)
    assert cb.eff(121.7817)/off.eff(121.7817) > 1.05
    assert cb._effcal_model == 'vidmar-5'
    assert cb._summing['L'] == 0.69 and cb._summing['xray_partners'] == pytest.approx(0.581, abs=5E-4)


@requires_data('decay')
def test_effcal_I_min_point_failing_both_cuts():
    # the 963.4 keV line (0.14%) fails the uncertainty cut too; below the floor
    # it carries the intensity reason, and without the floor the uncertainty one
    def reason(**kw):
        cb = ci.Calibration()
        cb.calibrate([eu_sp()], sources=EU_SOURCES, **kw)
        d = cb.effcal_data
        r = d.loc[np.isclose(d['energy'], 963.367), 'reason'].tolist()
        assert len(r) == 1
        return r[0], cb.diagnostics.set_index('fit').loc['effcal', 'message']
    r1, msg1 = reason()
    r0, msg0 = reason(effcal_I_min=0)
    assert (r1, r0) == ('intensity<1%', 'unc>33%')
    # the uncertainty clause counts only the points the floor kept
    assert 'unc>33% of value' not in msg1 and '(1 dropped: unc>33% of value)' in msg0


@requires_data('decay')
def test_effcal_I_min_leaves_user_points():
    # a floor of 10% keeps 7 measured points; the two user points carry no
    # line intensity and are never dropped by it, whatever the floor
    ep = pd.DataFrame({'energy': [2000.0, 2500.0], 'efficiency': [0.003, 0.0025],
                       'unc_efficiency': [0.0003, 0.00025], 'isotope': ['56CO', '56CO']})
    sp = eu_sp()
    inten = _line_intensities(sp)
    n_meas = sum(100.0*i >= 10.0 for i in inten.values())
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES, eff_points=ep, effcal_I_min=10, outlier_sigma=1E9)
    d = cb.effcal_data
    assert d.loc[d['isotope'] == '56CO', 'used'].tolist() == [True, True]
    assert int(cb.diagnostics.set_index('fit').loc['effcal', 'n_points']) == n_meas + 2 == 9
    assert cb._effcal_erange[1] == 2500.0


def test_effcal_I_min_validation():
    # a negative floor is refused by the entry's validation (it would read as
    # a floor yet keep every line); zero and positive numbers are accepted
    cb = ci.Calibration()
    assert cb.fit_config['effcal_I_min'] == 1
    for bad in [-1, -0.5, float('nan'), '1']:
        with pytest.raises(ValueError, match="'effcal_I_min' must be a non-negative number"):
            cb.fit_config = {'effcal_I_min': bad}
    with pytest.raises(ValueError, match="'effcal_I_min' must be a non-negative number"):
        cb.calibrate([], sources=EU_SOURCES, effcal_I_min=-1)
    for good in [0, 0.5, 2]:
        cb.fit_config = {'effcal_I_min': good}
        assert cb.fit_config['effcal_I_min'] == good


@requires_data('decay')
def test_effcal_data_floor_dropped_point():
    # a point the floor dropped is listed in effcal_data as not used, with the
    # floor in its reason, and is stored with the other dropped points (which
    # the efficiency plot draws as rejected points): the 488.7 keV line
    # (0.41%) below a floor of 0.5%, beside the 563.99 keV line (0.49%), also
    # below it, and the 688.67 keV line (0.86%), above it
    cb = ci.Calibration()
    cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=0.5)
    d = cb.effcal_data
    row = d[np.isclose(d['energy'], 488.6792)]
    assert len(row) == 1
    assert not row['used'].iloc[0] and row['reason'].iloc[0] == 'intensity<0.5%'
    assert np.isfinite(row['residual'].iloc[0])
    assert 488.68 in np.round(cb._calib_data['effcal_dropped']['energy'], 2)
    assert 488.68 not in np.round(cb._calib_data['effcal']['energy'], 2)
    assert d.loc[np.isclose(d['energy'], 563.986), 'reason'].tolist() == ['intensity<0.5%']
    assert d.loc[np.isclose(d['energy'], 688.67), 'reason'].tolist() == ['']


@requires_data('decay')
def test_effcal_I_min_keeps_line_at_floor(monkeypatch):
    # a line whose tabulated intensity equals the floor stays in the fit, as
    # the peak fit's I_min keeps it.  The 488.68 keV line (0.414% in the
    # library) is set to 0.41%, a value whose per-decay fraction 0.41*1E-2,
    # multiplied back by 100, falls below 0.41 in floating point
    import curie.isotope as iso_mod
    assert 100.0*(0.41*1E-2) < 0.41
    gammas = iso_mod.Isotope.gammas

    def gammas_488_at_041(self, *args, **kwargs):
        g = gammas(self, *args, **kwargs)
        if self.name == '152EUg':
            g.loc[np.isclose(g['energy'], 488.6792), 'intensity'] = 0.41
        return g
    monkeypatch.setattr(iso_mod.Isotope, 'gammas', gammas_488_at_041)

    def reason(floor):
        cb = ci.Calibration()
        cb.calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=floor, outlier_sigma=1E9)
        d = cb.effcal_data
        r = d.loc[np.isclose(d['energy'], 488.6792), 'reason'].tolist()
        assert len(r) == 1
        return r[0]
    assert reason(0.41) == ''
    assert reason(0.4100001) == 'intensity<0.41%'


# A calibration file written by curie at commit 617dcd7, before the floor
# existed: the example spectrum calibrated with default settings and saved with
# cb.saveas on 2026-11-10, kept here with its efficiency groups only.  The
# values below are what that curie evaluated from the file at full precision
PRE_FLOOR_CALIB = {
    'engcal': [0.2817511939530732, 0.18264563760053337, 3.529858557863143e-08],
    'effcal': [0.04726526076030443, 9.999999999999387, 1.9552472546405821, 1.8222490826810045,
        0.4265249102736406],
    'unc_effcal': [[6.0233839911486326e-05, 0.27083796362652945, 0.009976489387214946,
        0.005479547595603378, -0.0025716683151875623], [0.2708379636265295, 1830.097974944452,
        62.812719049870466, 26.031240014106057, -12.72454116556868], [0.009976489387214947,
        62.812719049870466, 2.1978960800208074, 0.9749909484322439, -0.4665457051357231],
        [0.005479547595603378, 26.031240014106057, 0.9749909484322439, 0.5449045983146338,
        -0.24771857684656098], [-0.0025716683151875623, -12.72454116556868, -0.4665457051357231,
        -0.24771857684656098, 0.11492550556664012]],
    'rescal': [2.2957297512503443, 0.0003551252597496196],
    'engcal_model': 'quadratic',
    'rescal_model': 'linear',
    'effcal_model': 'vidmar-5',
    'effcal_erange': [121.7817, 1457.643],
    '_calib_data': {
        'effcal': {
            'energy':
                [121.7817, 251.633, 271.08, 295.9387, 324.83, 329.41, 344.2785, 367.7891,
                411.1165, 416.02, 488.6792, 503.467, 563.986, 566.438, 586.2648, 656.489,
                674.64, 678.623, 688.67, 719.346, 764.88, 778.9045, 810.451, 841.574, 919.337,
                926.31, 964.057, 1084.0, 1085.837, 1089.737, 1112.076, 1212.948, 1249.94,
                1292.78, 1299.142, 1408.013, 1457.643],
            'efficiency':
                [0.03394129443222171, 0.01767470172829933, 0.02315538585971057,
                0.01878714957833821, 0.018292532162053825, 0.016883428485527655,
                0.015623961765579933, 0.016001212432561807, 0.012723501008202031,
                0.01022991109453062, 0.010280463133361856, 0.009282600191284303,
                0.009398259205190163, 0.009671457072753584, 0.008720405636831713,
                0.010487022996777316, 0.007228833875504719, 0.007288502542410405,
                0.00790675862717256, 0.007220200829576929, 0.007414837926193115,
                0.007026945328459143, 0.007328411808413339, 0.006149281986109922,
                0.006286781353852147, 0.005589981876599268, 0.0050579774715388305,
                0.0052467193005769665, 0.005312379089129848, 0.005135812649463597,
                0.004963053288997086, 0.004576146931565704, 0.005725180468001319,
                0.004022533676108932, 0.004359127788935042, 0.004035172604226635,
                0.0049629902174368095],
            'unc_efficiency':
                [0.0002925048845026896, 0.0032417472918329798, 0.002586789703953722,
                0.0004646332797342676, 0.001906568483611957, 0.0012970399439420981,
                0.00012666818308065524, 0.000573590652616743, 0.00013645684657046076,
                0.0011817230064845488, 0.0003703316984942743, 0.0007771724177327119,
                0.00032017705123706276, 0.0008961294065176484, 0.0006651827837875337,
                0.001104606281632887, 0.0007074390557197295, 0.00032479358795723236,
                0.0001934563558847226, 0.0005256064718461575, 0.0007752867022195295,
                6.236621197874485e-05, 0.0004350366620703712, 0.0007786714053274695,
                0.00035665249066788257, 0.00047817963177641565, 0.00041470292417849434,
                0.0008041308447776235, 4.845522387361092e-05, 0.00010702464565619279,
                4.1932652844438764e-05, 0.00010628415093605725, 0.0004371663211708215,
                0.0005795227785212679, 8.713124054452847e-05, 5.615195296496656e-05,
                0.0004422290092699715],
            'line':
                ['152EU:121.78', '152EU:251.63', '152EU:271.08', '152EU:295.94', '152EU:324.83',
                '152EU:329.41', '152EU:344.28', '152EU:367.79', '152EU:411.12', '152EU:416.02',
                '152EU:488.68', '152EU:503.47', '152EU:563.99', '152EU:566.44', '152EU:586.26',
                '152EU:656.49', '152EU:674.64', '152EU:678.62', '152EU:688.67', '152EU:719.35',
                '152EU:764.88', '152EU:778.90', '152EU:810.45', '152EU:841.57', '152EU:919.34',
                '152EU:926.31', '152EU:964.06', '152EU:1084.00', '152EU:1085.84',
                '152EU:1089.74', '152EU:1112.08', '152EU:1212.95', '152EU:1249.94',
                '152EU:1292.78', '152EU:1299.14', '152EU:1408.01', '152EU:1457.64'],
            'fit':
                [0.04726526076030443, 9.999999999999387, 1.9552472546405821, 1.8222490826810045,
                0.4265249102736406],
            'unc':
                [[6.0233839911486326e-05, 0.27083796362652945, 0.009976489387214946,
                0.005479547595603378, -0.0025716683151875623], [0.2708379636265295,
                1830.097974944452, 62.812719049870466, 26.031240014106057, -12.72454116556868],
                [0.009976489387214947, 62.812719049870466, 2.1978960800208074,
                0.9749909484322439, -0.4665457051357231], [0.005479547595603378,
                26.031240014106057, 0.9749909484322439, 0.5449045983146338,
                -0.24771857684656098], [-0.0025716683151875623, -12.72454116556868,
                -0.4665457051357231, -0.24771857684656098, 0.11492550556664012]],
        },
        'effcal_dropped': {
            'energy':
                [963.367, 244.6974, 443.9606, 867.38, 1005.27, 1109.18, 1528.1],
            'efficiency':
                [0.06712395765468603, 0.020620880652903757, 0.011488009918483212,
                0.006077674728375348, 0.007997996902232017, 0.009254736279126758,
                0.008547756632893349],
            'unc_efficiency':
                [0.04281628379381938, 0.0001503733329665915, 0.00011125306614568644,
                7.654251232956714e-05, 0.0003006683535100719, 0.0006639238148955666,
                0.000499085473068489],
            'line':
                ['152EU:963.37', '152EU:244.70', '152EU:443.96', '152EU:867.38',
                '152EU:1005.27', '152EU:1109.18', '152EU:1528.10'],
            'reason':
                ['unc>33%', 'outlier >3.16 sigma', 'outlier >3.16 sigma', 'outlier >3.16 sigma',
                'outlier >3.16 sigma', 'outlier >3.16 sigma', 'outlier >3.16 sigma'],
        },
    },
}


PRE_FLOOR_EVAL = {
    'E': [100.0, 121.7817, 344.2785, 661.657, 1408.013, 2000.0],
    'eff': [0.03705507143289238, 0.033906821420302095, 0.01527646088189391, 0.008328509870778938,
            0.004001890754711425, 0.0027744986975451655],
    'unc_eff': [0.0027701076288391205, 0.0012097237137524608, 0.00021223100679876452,
                0.00024979538726836915, 0.00015517760593363556, 0.0002518010999996607],
    'eng': [182.9626873800651, 914.3924038360857],
    'res': [2.650855010999964, 4.0713560499984425],
}


def test_pre_floor_calibration_file_unchanged(tmp_path):
    # a file saved before the floor existed loads and evaluates as that curie
    # evaluated it: the floor acts in calibrate() only, never on loading
    f = str(tmp_path / 'pre_floor.json')
    json.dump(PRE_FLOOR_CALIB, open(f, 'w'))
    cb = ci.Calibration(f)
    ev = PRE_FLOOR_EVAL
    assert (cb._engcal_model, cb._rescal_model, cb._effcal_model) == ('quadratic', 'linear', 'vidmar-5')
    assert cb._effcal_erange == (121.7817, 1457.643)
    np.testing.assert_allclose(cb.eff(np.array(ev['E'])), ev['eff'], rtol=1E-12)
    np.testing.assert_allclose(cb.unc_eff(np.array(ev['E'])), ev['unc_eff'], rtol=1E-12)
    np.testing.assert_allclose(cb.eng(np.array([1000, 5000])), ev['eng'], rtol=1E-12)
    np.testing.assert_allclose(cb.res(np.array([1000, 5000])), ev['res'], rtol=1E-12)
    d = cb.effcal_data
    assert d['reason'].tolist() == ['']*37 + ['unc>33%'] + ['outlier >3.16 sigma']*6
    np.testing.assert_allclose(d['residual'], d['efficiency']-cb.eff(d['energy'].to_numpy()), rtol=1E-12)


# Synthetic 152Eu spectrum from a known efficiency curve: every 152Eu gamma
# line from 75 to 2000 keV as a peak of the shape the peak fit uses (Gaussian
# with the fixed low-energy tail R=0.1, alpha=0.9) on a flat continuum, at a
# binning and resolution typical of a coaxial HPGe, then one Poisson
# realization.  The source decays are counted from its reference date, so each
# line's area is eff(E)*I*A0*(1-exp(-lambda*t))/lambda exactly
SYN_EFF = [0.045, 6.0, 2.0, 1.8, 0.4]   # vidmar-5 parameters, inside the fit bounds
SYN_KEV_PER_CH, SYN_N_CH = 0.25, 8192
SYN_RESCAL = [1.87, 2.7E-4]            # sigma 0.5 keV at 122 keV, 0.85 keV at 1408 keV
SYN_A0, SYN_T, SYN_CONT = 1E5, 3600.0, 200.0
SYN_REF = '01/01/2020 00:00:00'
# the weak line (0.856%) and a peak beside it that is not on the fit list, 0.4
# keV below it (0.6 resolution widths) with as many counts as the weak line
# itself, so the fitted peak holds about twice the line's counts - as a
# room-background line beside a weak 152Eu line does
SYN_WEAK_E, SYN_INTERFERER_DE = 688.670, -0.4


def _synthetic_eu(interferer=True, seed=1):
    import datetime as dtm
    ip = ci.Isotope('152EU')
    n_dec = SYN_A0*(1.0-np.exp(-ip.dc*SYN_T))/ip.dc
    ch = np.arange(SYN_N_CH).astype(float)
    shape = ci.Spectrum()
    R, alpha = shape.fit_config['R'], shape.fit_config['alpha']
    lam = np.full(SYN_N_CH, SYN_CONT)

    def add(counts, energy):
        mu = energy/SYN_KEV_PER_CH
        sig = SYN_RESCAL[0]+SYN_RESCAL[1]*mu
        amp = counts/(2.506628*sig + 2*R*alpha*sig*np.exp(-0.5/alpha**2))
        w = np.abs(ch-mu) < 30*sig
        lam[w] += shape._peak(ch[w], amp, mu, sig, R, alpha, 0.0)

    weak = None
    for E, I in zip(*ip.gammas(E_lim=[75.0, 2000.0])[['energy', 'intensity']].to_numpy().T):
        counts = float(ci.Calibration().eff(E, SYN_EFF))*I*1E-2*n_dec
        add(counts, E)
        if abs(E-SYN_WEAK_E) < 1E-3:
            weak = counts
    if interferer:
        add(weak, SYN_WEAK_E+SYN_INTERFERER_DE)
    sp = ci.Spectrum()
    sp.hist = np.random.default_rng(seed).poisson(lam).astype(np.int64)
    sp.cb.engcal = [0.0, SYN_KEV_PER_CH, 0.0]
    sp.cb.rescal = SYN_RESCAL
    sp.start_time = dtm.datetime.strptime(SYN_REF, '%m/%d/%Y %H:%M:%S')
    sp.live_time = sp.real_time = SYN_T
    sp._snip_bg()
    sp.isotopes = ['152EU']
    return sp, weak


@pytest.mark.validation
@requires_data('decay')
def test_effcal_I_min_recovers_known_curve_beside_interfered_line():
    # the known curve is the truth; the tolerance is 1% at five energies from
    # 245 to 1112 keV, 661.657 keV among them.  It is derived from the
    # precision of the default fit: the 14 lines of at least 1% carry
    # counting uncertainties with a median below 0.5% (asserted; 0.34% here),
    # and the spectrum is generated from the tabulated intensities themselves,
    # so their uncertainties do not scatter these points.  A curve resting on
    # them departs from the truth by a few tenths of a percent: 0.2% at most
    # on this realization and 0.6% at most over seeds 1 to 10, so 1% holds
    # with margin and is the agreement the default selection is required to
    # give at 661.657 keV.  The interfered line reads about 2 times high,
    # which pulls the fit with every line in it by 3.1% at 661.657 keV (3.1
    # to 5.5% over the same seeds); without the interferer that fit recovers
    # the curve within 0.7%, so the departure is the interfered line's
    src = [{'isotope': '152EU', 'A0': SYN_A0, 'ref_date': SYN_REF}]
    E = np.array([244.6974, 344.2785, 661.657, 778.9045, 1112.076])
    truth = ci.Calibration().eff(E, SYN_EFF)

    def departure(interferer, **kw):
        sp, weak = _synthetic_eu(interferer)
        pk = sp.peaks
        cb = ci.Calibration()
        cb.calibrate([sp], sources=src, summing=False, **kw)
        return cb.eff(E)/truth-1.0, pk, weak

    dev, pk, weak = departure(True)
    strong = pk[pk['intensity'] >= 0.01]
    assert np.median(strong['unc_counts']/strong['counts']) < 0.005
    w = pk[np.isclose(pk['energy'], SYN_WEAK_E)].iloc[0]
    assert w['intensity'] < 0.01 and 1.8 < w['counts']/weak < 2.2
    assert np.all(np.abs(dev) < 0.01), dev
    dev0, _, _ = departure(True, effcal_I_min=0)
    assert abs(dev0[2]) > 0.01, dev0
    dev0_clean, _, _ = departure(False, effcal_I_min=0)
    assert np.all(np.abs(dev0_clean) < 0.01), dev0_clean


def _eu_points_by_intensity():
    # the 152Eu efficiency points of the example spectrum and the number whose
    # line intensity is below 20% and 30% (all of them: the strongest line,
    # 121.8 keV, is 28.5%)
    inten = list(_line_intensities(eu_sp()).values())
    return len(inten), {f: sum(100.0*i < f for i in inten) for f in (20, 30)}


@requires_data('decay')
def test_effcal_I_min_all_points_below_floor():
    # every line below the floor: the all-points-dropped error counts what the
    # floor dropped and names effcal_I_min as the setting to lower
    n, below = _eu_points_by_intensity()
    assert below[30] == n
    with pytest.raises(ValueError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=30)
    msg = str(ei.value)
    assert msg == ('Calibration.calibrate: all {0} effcal points dropped - cannot fit the efficiency calibration. '
                   'Of {0} measured points, the line-intensity floor (intensity<30%) dropped {0} and the '
                   'uncertainty filter (unc>33% of value) 0. Lower effcal_I_min to keep weaker lines, and check the peak fits and source '
                   'activities.'.format(n))


@requires_data('decay')
@pytest.mark.parametrize('model,tag', [(None, 'vidmar-5'), ('loglog', 'loglog-4')])
def test_effcal_I_min_too_few_points(model, tag):
    # a floor of 20% leaves three lines (121.8, 344.3, 1408.0 keV): the
    # too-few-points error states what the floor dropped and names effcal_I_min
    n, below = _eu_points_by_intensity()
    assert n-below[20] == 3
    with pytest.raises(ValueError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=20, effcal_model=model)
    msg = str(ei.value)
    head = 'Calibration.calibrate: effcal_model {0} needs at least 5 efficiency points to fit (have 3)'.format(tag)
    assert msg == (head+'. Of {1} measured points, the line-intensity floor (intensity<20%) dropped {0} and the '
                   'uncertainty filter (unc>33% of value) 0. Lower effcal_I_min to keep weaker lines, check '
                   'the energy calibration and the peak fits, or {2}loosen the filters.'.format(
                       below[20], n, 'lower the order or ' if model == 'loglog' else ''))


# the efficiency-fit stop messages of the example spectrum before the floor
# existed, recorded 2026-11-10 at commit 617dcd7 from the calibrate() calls of
# test_effcal_stop_messages_where_floor_drops_none without effcal_I_min
PRE_FLOOR_STOPS = {
    'all_dropped': 'Calibration.calibrate: all 44 effcal points dropped (unc>0.0001% of value) - cannot fit the '
                   'efficiency calibration. Check the peak fits and source activities.',
    'vidmar-5': 'Calibration.calibrate: effcal_model vidmar-5 needs at least 5 efficiency points to fit (have 3); '
                'the uncertainty filter (unc>0.8% of value) dropped 43 of 44 measured points - check the energy '
                'calibration and the peak fits, or loosen the filters.',
    'loglog': 'Calibration.calibrate: effcal_model loglog-4 needs at least 5 efficiency points to fit (have 3); '
              'the uncertainty filter (unc>0.8% of value) dropped 43 of 44 measured points - check the energy '
              'calibration and the peak fits, or lower the order or loosen the filters.',
}


@requires_data('decay')
@pytest.mark.parametrize('floor', [0, 0.05])
@pytest.mark.parametrize('case', ['all_dropped', 'vidmar-5', 'loglog'])
def test_effcal_stop_messages_where_floor_drops_none(case, floor):
    # where the floor drops no point - it is 0, or 0.05%, below every line the
    # peak fit keeps (its own I_min is 0.05%) - each stop message is the text
    # it had before the floor existed
    ep = pd.DataFrame([
        {'isotope': 'user', 'energy': 50.0, 'efficiency': 0.0060, 'unc_efficiency': 0.0003},
        {'isotope': 'user', 'energy': 90.0, 'efficiency': 0.0075, 'unc_efficiency': 0.0004}])
    kw = {'all_dropped': dict(effcal_max_error=1E-6),
          'vidmar-5': dict(effcal_model='vidmar-5', effcal_max_error=0.008, outlier_sigma=1E9, eff_points=ep),
          'loglog': dict(effcal_model='loglog', effcal_max_error=0.008, outlier_sigma=1E9, eff_points=ep)}[case]
    with pytest.raises(ValueError) as ei:
        ci.Calibration().calibrate([eu_sp()], sources=EU_SOURCES, effcal_I_min=floor, **kw)
    assert str(ei.value) == PRE_FLOOR_STOPS[case]
