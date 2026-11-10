"""Tests of the coincidence-summing correction as calibrate() and Spectrum apply it.

The data-layer tests isolate through CURIE_DATA_DIR, a monkeypatched registry and
fetch seam (`data._retrieve`) and small fixture databases built on the fly; they
never touch the network or the real data directory.  The calibrate() tests use the
example spectrum and synthetic spectra built from a known efficiency curve and the
summing factors of `curie._tcs`.
"""
import contextlib
import copy
import datetime as dtm
import hashlib
import io
import sqlite3

import numpy as np
import pandas as pd
import pytest

import curie as ci
import curie.calibration as cal_mod
from curie import _tcs, data
from conftest import EXAMPLES_DIR, requires_data, requires_pairs


GENERATION = 'v7'


def _decay_db(path, generation=None, coincidences=True):
    """A decay database with an optional generation stamp and pair table."""
    con = sqlite3.connect(str(path))
    con.execute('CREATE TABLE chart (isotope TEXT, name TEXT)')
    if generation is not None:
        con.execute('CREATE TABLE _version (library TEXT, generation TEXT)')
        con.execute("INSERT INTO _version VALUES ('decay', ?)", (generation,))
    if coincidences:
        con.execute('CREATE TABLE coincidences (isotope TEXT, isomer TEXT, energy REAL, coinc_energy REAL, '
                    'coinc_type TEXT, intensity REAL, unc_intensity REAL)')
        con.execute("INSERT INTO coincidences VALUES ('60CO', 'g', 1173.228, 1332.492, 'g', 99.8, NULL)")
    con.commit()
    con.close()
    with open(str(path), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def datadir(tmp_path, monkeypatch):
    """An isolated data directory, registry of generation v7, fetch seam and connection cache."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    (tmp_path / 'remote').mkdir()
    monkeypatch.setenv('CURIE_DATA_DIR', str(cache))
    monkeypatch.setattr(data, '_site_data_paths', lambda db='': [str(tmp_path / 'site' / db)])
    monkeypatch.setattr(data, '_legacy_data_path', lambda db='': str(tmp_path / 'legacy' / db))
    registry = {'base_url': 'https://example.invalid/releases/download/{}/'.format(GENERATION), 'files': {}, 'shards': {}}
    monkeypatch.setattr(data, '_registry', lambda: registry)
    monkeypatch.setattr(data, 'GLOB_CONNECTIONS_DICT', {})

    def fake_retrieve(fnm):
        dest = cache / fnm
        dest.write_bytes((tmp_path / 'remote' / fnm).read_bytes())
        return str(dest)

    monkeypatch.setattr(data, '_retrieve', fake_retrieve)
    return {'tmp': tmp_path, 'cache': cache, 'remote': tmp_path / 'remote', 'registry': registry}


@pytest.mark.parametrize('generation, coincidences, usable, warning', [
    (None, True, False, 'predates the generation stamp'),
    ('v6', True, False, 'is from data generation v6'),
    (GENERATION, True, True, None),
    (GENERATION, False, False, None),
], ids=['no-stamp', 'other-generation', 'expected-generation', 'expected-generation-no-pair-table'])
def test_pair_table_usable(datadir, capsys, generation, coincidences, usable, warning):
    _decay_db(datadir['cache'] / 'decay.db', generation, coincidences)
    assert data._pair_table_usable() is usable
    out = capsys.readouterr().out
    if warning is None:
        assert 'generation' not in out
    else:
        assert warning in out and "ci.download('decay')" in out
    # asked again: the same answer from the same connection, and no second warning
    assert data._pair_table_usable() is usable
    assert 'generation' not in capsys.readouterr().out


def test_pair_table_usable_follows_the_file(datadir, monkeypatch):
    # a stale file and a current one, each in its own data directory
    other = datadir['tmp'] / 'other'
    other.mkdir()
    _decay_db(datadir['cache'] / 'decay.db', 'v6')
    _decay_db(other / 'decay.db', GENERATION)
    assert data._pair_table_usable() is False
    monkeypatch.setenv('CURIE_DATA_DIR', str(other))
    assert data._pair_table_usable() is True
    monkeypatch.setenv('CURIE_DATA_DIR', str(datadir['cache']))
    assert data._pair_table_usable() is False

    # the stale file replaced by a re-download in the same session
    datadir['registry']['files']['decay.db'] = _decay_db(datadir['remote'] / 'decay.db', GENERATION)
    data.download('decay')
    assert data._pair_table_usable() is True


# ---------------------------------------------------------------------------
# calibrate(): the summing correction of the efficiency fit

EU_SOURCES = [{'isotope': '152EU', 'A0': 3.7E4, 'ref_date': '01/01/2009 12:00:00'}]
E_CURVE = np.array([121.7817, 344.2785, 778.9045, 1408.013])

# The example spectrum calibrated by the parent of the summing correction with
# default settings, recorded 2026-11-10 at commit 1040f36: the curve and its
# fitted uncertainty at E_CURVE, the model tag and the calibrate() log lines.
# The curve tolerance holds across scipy/BLAS stacks (observed spread 2E-6);
# the correction moves it by 4 to 6% here
PARENT_EFF = [0.03389452960116718, 0.015288308693332031, 0.007129342928757225, 0.003908902030484131]
PARENT_UNC_EFF = [0.0009457745016816501, 0.00017719110217568457, 0.00017782177587114128, 0.00012232305549005895]
PARENT_LOG = [
    '[INFO] Calibration.calibrate: engcal [quadratic] fit to 44/44 points; chi2/dof=44',
    '[INFO] Calibration.calibrate: rescal [linear] fit to 44/44 points; 4 outliers: residual >3.16 sigma; chi2/dof=3.7',
    '[INFO] Calibration.calibrate: effcal model: vidmar-5 (auto - no x-ray peaks fit, fewer than two points below 150 keV)',
    '[INFO] Calibration.calibrate: effcal [vidmar-5] fit to 14/44 points; 30 dropped: line intensity <1% (effcal_I_min); '
    '3 outliers: residual >3.16 sigma [152EU 444.0, 152EU 867.4, 152EU 1085.8 keV]; chi2/dof=7.9; '
    'independent uncertainty components scaled x4.21',
]


@pytest.fixture
def pairs_usable():
    """Skip unless the installed decay database passes the generation check."""
    if not data._pair_table_usable():
        pytest.skip('the installed decay database fails the data-generation check')


@pytest.fixture(scope='module')
def eu_fitted():
    """The example spectrum (152EU, 40K) with its peaks fitted once."""
    sp = ci.Spectrum(str(EXAMPLES_DIR / 'eu_calib_7cm.Spe'))
    sp.isotopes = ['152EU', '40K']
    sp.fit_peaks()
    return sp


def _calibrate(spectra, sources, capsys=None, **kw):
    """calibrate() on copies of fitted spectra; returns the calibration and its log lines."""
    if capsys is not None:
        capsys.readouterr()
    cb = ci.Calibration()
    cb.calibrate([copy.deepcopy(sp) for sp in spectra], sources=sources, **kw)
    out = capsys.readouterr().out if capsys is not None else ''
    return cb, [l for l in out.splitlines() if 'Calibration.calibrate' in l]


def _apparent(sp, sources):
    """The measured efficiency of each source peak of a fitted spectrum, by its line label."""
    out = {}
    for _, pk in sp.peaks.iterrows():
        src = [s for s in sources if s['isotope'] == pk['isotope']]
        if not src:
            continue
        ip = ci.Isotope(pk['isotope'])
        td = (sp.start_time-pd.to_datetime(src[0]['ref_date'], format='%m/%d/%Y %H:%M:%S')).total_seconds()
        eff = pk['counts']*ip.dc/(sp._geom_corr*(1.0-np.exp(-ip.dc*sp.real_time))*np.exp(-ip.dc*td)
                                  * pk['intensity']*src[0]['A0']*(sp.live_time/sp.real_time))
        out[pk['isotope']+':'+'{:.2f}'.format(pk['energy'])] = (pk['isotope'], pk['energy'], eff)
    return out


def _stored(cb):
    """Every stored efficiency point, used or dropped: {line: (efficiency, unc_efficiency, group)}."""
    out = {}
    for grp in ('effcal', 'effcal_dropped'):
        d = cb._calib_data[grp]
        for ln, e, u in zip(d['line'], d['efficiency'], d['unc_efficiency']):
            assert ln not in out
            out[str(ln)] = (float(e), float(u), grp if grp == 'effcal' else
                            str(d['reason'][list(d['line']).index(ln)]))
    return out


# synthetic detector: the binning, resolution and peak shape of the synthetic
# spectrum of tests/test_calibration_models.py, and the vidmar-5 curve
SYN_EFF = [0.045, 6.0, 2.0, 1.8, 0.4]
SYN_KEV_PER_CH, SYN_N_CH = 0.25, 8192
SYN_RESCAL = [1.87, 2.7E-4]
SYN_A0, SYN_T, SYN_CONT = 1E5, 3600.0, 200.0
SYN_REF = '01/01/2020 00:00:00'
SYN_L = 0.69


def _true_factors(nuclide, energy, scale):
    """Summing factors at the synthetic curve, from the array form of curie._tcs;
    1 for a line without a factor."""
    arr = _tcs._arrays(nuclide)
    ep = ci.Calibration().eff(arr.grid, SYN_EFF)
    F, _ = arr.terms(ep, ep/cal_mod._peak_to_total(arr.grid, SYN_L), scale)
    j = arr.index(energy)
    return np.nan_to_num(np.where(j >= 0, F[j], 1.0), nan=1.0)


def _synthetic(nuclides, scale, kx_ratio=None, live=SYN_T, dip=False, seed=1):
    """A spectrum of the nuclides (each SYN_A0 at the reference date) whose peak areas are
    the synthetic curve times the summing factor of each line at the x-ray partner scale
    `scale`, on a flat continuum, one Poisson realization.  kx_ratio places Sm K-alpha
    counts of that ratio to the 121.78 keV area; dip empties the continuum inside the
    Sm K-alpha window, so the window reads below its sidebands."""
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

    area = {}
    for nm in nuclides:
        ip = ci.Isotope(nm)
        n_dec = SYN_A0*(1.0-np.exp(-ip.dc*live))/ip.dc
        g = ip.gammas(E_lim=[75.0, 2000.0])
        E, I = g['energy'].to_numpy(dtype=float), g['intensity'].to_numpy(dtype=float)
        for e, i, f in zip(E, I, _true_factors(nm, E, scale)):
            area[(nm, e)] = float(ci.Calibration().eff(e, SYN_EFF))*i*1E-2*n_dec*f
            add(area[(nm, e)], e)
    if kx_ratio is not None:
        # K-alpha2 and K-alpha1 in the ratio 0.55 : 1
        W = kx_ratio*area[('152EU', 121.7817)]
        add(W*0.55/1.55, 39.522)
        add(W/1.55, 40.118)
    if dip:
        # the window spans about 37.8 to 42.0 keV, its sidebands about 1.1
        # keV on either side
        lam[int(38.0/SYN_KEV_PER_CH):int(41.75/SYN_KEV_PER_CH)] = 0.0
    sp = ci.Spectrum()
    sp.hist = np.random.default_rng(seed).poisson(lam).astype(np.int64)
    sp.cb.engcal = [0.0, SYN_KEV_PER_CH, 0.0]
    sp.cb.rescal = SYN_RESCAL
    sp.start_time = dtm.datetime.strptime(SYN_REF, '%m/%d/%Y %H:%M:%S')
    sp.live_time = sp.real_time = live
    sp._snip_bg()
    sp.isotopes = list(nuclides)
    with contextlib.redirect_stdout(io.StringIO()):
        sp.fit_peaks()
    return sp


def _sources(nuclides):
    return [{'isotope': nm, 'A0': SYN_A0, 'ref_date': SYN_REF} for nm in nuclides]


SYN_SCALE = 0.6


@pytest.fixture(scope='module')
def eu_synthetic():
    """The synthetic 152Eu spectrum at x-ray partner scale 0.6, with the Sm K-alpha
    counts of that scale."""
    return _synthetic(['152EU'], SYN_SCALE, kx_ratio=SYN_SCALE*cal_mod._KX_RATIO_FULL)


@pytest.mark.validation
@requires_data('decay')
@requires_pairs()
def test_summing_recovers_known_curve(eu_synthetic, pairs_usable, capsys):
    # the known curve and scale are the truth.  The 1% tolerance is that of
    # the summing-free synthetic test in tests/test_calibration_models.py; the
    # corrected fit departs from the truth by 0.69% at most on this
    # realization and 0.94% at most over seeds 1 to 6, and recovers the scale
    # within 0.2%.  Fitted without the correction, the curve lies below the
    # truth by 3.5 to 6.1% at these energies, where the factors of the lines
    # that fit keeps (0.93 to 0.97) average 0.954
    E = np.array([244.6974, 344.2785, 661.657, 778.9045, 1112.076])
    truth = ci.Calibration().eff(E, SYN_EFF)
    cb, log = _calibrate([eu_synthetic], _sources(['152EU']), capsys)
    dev = cb.eff(E)/truth-1.0
    assert np.all(np.abs(dev) < 0.01), dev
    assert cb._summing['L'] == SYN_L
    assert abs(cb._summing['xray_partners']/SYN_SCALE-1.0) < 0.05
    # the INFO line states the scale and the largest correction of a fit
    # point (the points kept and those the outlier screen flagged)
    info = [l for l in log if 'coincidence-summing correction applied' in l]
    assert len(info) == 1
    assert 'x-ray partner scale {0:.3g} from 1 152EU spectrum'.format(cb._summing['xray_partners']) in info[0]
    d = cb.effcal_data
    fit_E = d.loc[d['used'] | d['reason'].str.startswith('outlier'), 'energy'].to_numpy()
    F = cb._summing_factors(['152EU']*len(fit_E), fit_E, cb.effcal, cb._summing['xray_partners'])[0]
    i = int(np.argmax(np.abs(F-1.0)))
    assert 'largest correction of a fit point {0:.2%} (152EU {1:.1f} keV, factor {2:.4f})'.format(
        abs(F[i]-1.0), fit_E[i], F[i]) in info[0]

    # without the correction, the curve departs from the truth by the size of
    # the correction: below it, by more than half and less than twice the
    # mean departure of the fit lines' factors from 1
    off, _ = _calibrate([eu_synthetic], _sources(['152EU']), summing=False)
    dev_off = off.eff(E)/truth-1.0
    mean_c = np.mean(_true_factors('152EU', off._calib_data['effcal']['energy'], SYN_SCALE)-1.0)
    assert mean_c < -0.02
    assert np.all((dev_off < 0.5*mean_c) & (dev_off > 2.0*mean_c)), (dev_off, mean_c)
    assert off._summing is None


@requires_data('decay')
@requires_pairs()
def test_summing_scale_pools_the_152eu_spectra(pairs_usable):
    # two 152Eu spectra whose window ratios read 0.2 and 1.4 of the fully
    # recording value, the second with three times the 121.78 keV counts: the
    # window sums and the 121.78 keV counts are each summed before the ratio,
    # which gives (0.2 + 3*1.4)/4 = 1.1, against 0.8 for the mean of the two
    # ratios and 0.2 or 1.4 for either spectrum alone
    full = cal_mod._KX_RATIO_FULL
    a = _synthetic(['152EU'], 1.1, kx_ratio=0.2*full)
    b = _synthetic(['152EU'], 1.1, kx_ratio=1.4*full, live=3*SYN_T, seed=2)
    cb, _ = _calibrate([a, b], _sources(['152EU']))
    assert abs(cb._summing['xray_partners']/1.1-1.0) < 0.05


@requires_data('decay')
@requires_pairs()
def test_summing_window_below_sidebands_gives_scale_zero(pairs_usable, capsys):
    sp = _synthetic(['152EU'], 0.0, dip=True)
    cb, log = _calibrate([sp], _sources(['152EU']), capsys)
    assert cb._summing['xray_partners'] == 0.0
    assert any('x-ray partner scale 0 from 1 152EU spectrum' in l for l in log)


@requires_data('decay')
@requires_pairs()
def test_summing_set_without_152eu(pairs_usable, capsys):
    # no 152Eu spectrum: x-ray partners off (scale 0), and the INFO line says
    # so.  137Cs has no coincidence pairs, so its point keeps factor 1: it is
    # stored as measured, as with summing=False, while the 60Co points are
    # corrected
    nuc = ['133BA', '137CS', '60CO']
    sp = _synthetic(nuc, 0.0)
    cb, log = _calibrate([sp], _sources(nuc), capsys)
    assert cb._summing['xray_partners'] == 0.0
    info = [l for l in log if 'coincidence-summing correction applied' in l]
    assert len(info) == 1 and 'x-ray partners off, as no spectrum has a fitted 152EU 121.8 keV peak' in info[0]
    assert len(_tcs._pairs('137CS')) == 0
    off, _ = _calibrate([sp], _sources(nuc), summing=False)
    on_pts, off_pts = _stored(cb), _stored(off)
    assert on_pts['137CS:661.66'][:2] == off_pts['137CS:661.66'][:2]
    for ln in ('60CO:1173.23', '60CO:1332.49'):
        assert abs(on_pts[ln][0]/off_pts[ln][0]-1.0) > 0.01


@requires_data('decay')
def test_summing_off_is_the_parent(eu_fitted, capsys, monkeypatch):
    # summing=False runs the parent's code path: no factor, scale or table
    # check is reached, and the result is the parent's
    def unreachable(*a, **k):
        raise AssertionError('summing code reached with summing=False')
    monkeypatch.setattr(cal_mod.Calibration, '_summing_factors', unreachable)
    monkeypatch.setattr(cal_mod.Calibration, '_xray_scale', unreachable)
    monkeypatch.setattr(cal_mod, '_pair_table_usable', unreachable)
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys, summing=False)
    np.testing.assert_allclose(cb.eff(E_CURVE), PARENT_EFF, rtol=1E-5)
    np.testing.assert_allclose(cb.unc_eff(E_CURVE), PARENT_UNC_EFF, rtol=1E-4)
    assert cb._effcal_model == 'vidmar-5' and cb._summing is None
    assert log == PARENT_LOG
    # every stored point is the measured value, as the parent stores it
    app = _apparent(eu_fitted, EU_SOURCES)
    pts = _stored(cb)
    assert set(pts) == set(app)
    for ln, (e, u, grp) in pts.items():
        assert e == pytest.approx(app[ln][2], rel=1E-12)


@requires_data('decay')
@requires_pairs()
def test_summing_stored_points_are_measured_over_factor(eu_fitted, pairs_usable):
    # kept, outlier-flagged and filter-dropped points each store the measured
    # value over its factor at the final curve; a tight outlier screen flags
    # some points
    cb, _ = _calibrate([eu_fitted], EU_SOURCES, outlier_sigma=1.5)
    app = _apparent(eu_fitted, EU_SOURCES)
    pts = _stored(cb)
    groups = {grp.split(' ')[0] for _, _, grp in pts.values()}
    assert groups == {'effcal', 'outlier', 'intensity<1%'}
    lines = sorted(pts)
    F, _, _ = cb._summing_factors([app[ln][0] for ln in lines], [app[ln][1] for ln in lines],
                                  cb.effcal, cb._summing['xray_partners'])
    off, _ = _calibrate([eu_fitted], EU_SOURCES, outlier_sigma=1.5, summing=False)
    off_pts = {ln: v for ln, v in _stored(off).items()}
    for ln, f in zip(lines, F):
        assert f != 1.0
        assert pts[ln][0] == pytest.approx(app[ln][2]/f, rel=1E-12)
        # the relative uncertainty is the measured one
        assert pts[ln][1]/pts[ln][0] == pytest.approx(off_pts[ln][1]/off_pts[ln][0], rel=1E-12)


@requires_data('decay')
@requires_pairs()
def test_summing_user_point_keeps_value(eu_fitted, pairs_usable):
    # a user point keeps its value and uncertainty, even with the label and
    # energy of a 152Eu line (user points take no factor, by position)
    ep = pd.DataFrame({'energy': [344.2785], 'efficiency': [0.0158], 'unc_efficiency': [0.0004],
                       'isotope': ['152EU']})
    cb, _ = _calibrate([eu_fitted], EU_SOURCES, eff_points=ep)
    d = cb.effcal_data
    user = d[(d['energy'] == 344.2785) & (d['efficiency'] == 0.0158)]
    assert len(user) == 1 and user.iloc[0]['unc_efficiency'] == 0.0004
    # the measured 344.28 keV point is corrected
    meas = d[(d['energy'] == 344.2785) & (d['efficiency'] != 0.0158)]
    assert len(meas) == 1
    assert meas.iloc[0]['efficiency'] != pytest.approx(_apparent(eu_fitted, EU_SOURCES)['152EU:344.28'][2], rel=1E-6)


@requires_data('decay')
@requires_pairs()
def test_summing_guarded_line(eu_fitted, pairs_usable, capsys, monkeypatch):
    # pair rows in which a partner of the 152Eu 344.28 keV line is three times
    # as intense as the line: that line keeps factor 1, and one warning names it
    real = _tcs._pairs
    g = ci.Isotope('152EU').gammas()
    I_344 = float(g.loc[g['energy'] == 344.2785, 'intensity'].iloc[0])

    def pairs(isotope):
        p = real(isotope)
        if isotope.name == '152EUg':
            on = (p['energy'] == 344.2785)&(p['coinc_type'] == 'g')
            p.loc[p[on]['intensity'].idxmax(), 'intensity'] = 3.0*I_344
        return p
    monkeypatch.setattr(_tcs, '_pairs', pairs)
    # arrays derived from the pair rows are kept with the connection: a store
    # for this test that holds only the outcome of the generation check
    con = data._get_connection('decay')
    monkeypatch.setattr(con, 'cache', {k: v for k, v in con.cache.items() if k in ('generation_ok', 'pair_table_usable')})
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys)
    warn = [l for l in log if 'no coincidence-summing correction for' in l]
    assert warn == ['[WARNING] Calibration.calibrate: no coincidence-summing correction for 152EU 344.3 keV: in the '
                    'decay data a coincidence pair of each of these lines is more intense than the line itself.']
    pts = _stored(cb)
    assert pts['152EU:344.28'][0] == pytest.approx(_apparent(eu_fitted, EU_SOURCES)['152EU:344.28'][2], rel=1E-12)
    assert pts['152EU:778.90'][0] != pytest.approx(_apparent(eu_fitted, EU_SOURCES)['152EU:778.90'][2], rel=1E-6)


def _same_result(a, b):
    np.testing.assert_array_equal(a.effcal, b.effcal)
    np.testing.assert_array_equal(a.unc_effcal, b.unc_effcal)
    assert a._effcal_model == b._effcal_model
    assert _stored(a) == _stored(b)


@requires_data('decay')
@requires_pairs()
def test_summing_stale_table_and_loglog(eu_fitted, capsys, monkeypatch):
    # a pair table that may not be used, and a loglog model, each give the
    # result of summing=False, no entry, and their message
    off, log_off = _calibrate([eu_fitted], EU_SOURCES, capsys, summing=False)
    monkeypatch.setitem(data._get_connection('decay').cache, 'pair_table_usable', False)
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys)
    _same_result(cb, off)
    assert cb._summing is None
    assert log == log_off[:2]+["[WARNING] Calibration.calibrate: calibrate() applies no coincidence-summing correction, "
                               "because the decay database failed the data-generation check or has no coincidences table. Run "
                               "ci.download('decay') to update it."]+log_off[2:]
    monkeypatch.undo()

    off, log_off = _calibrate([eu_fitted], EU_SOURCES, capsys, summing=False, effcal_model='loglog')
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys, effcal_model='loglog')
    _same_result(cb, off)
    assert cb._summing is None
    assert log == log_off[:2]+['[WARNING] Calibration.calibrate: the coincidence-summing correction needs a Vidmar '
                               'efficiency model, so calibrate() fits the loglog-4 curve without it. Set summing=False '
                               'to fit it without this warning.']+log_off[2:]


def test_summing_config_is_a_bool():
    cb = ci.Calibration()
    assert cb.fit_config['summing'] is True
    with pytest.raises(ValueError, match="'summing' must be a bool"):
        cb.fit_config = {'summing': 'yes'}
    cb.fit_config = {'summing': False}
    assert cb.fit_config['summing'] is False


@requires_data('decay')
@requires_pairs()
def test_summing_vidmar_7(eu_fitted, pairs_usable):
    # a user-selected vidmar-7 fits with the correction
    cb, _ = _calibrate([eu_fitted], EU_SOURCES, effcal_model='vidmar-7')
    off, _ = _calibrate([eu_fitted], EU_SOURCES, effcal_model='vidmar-7', summing=False)
    assert cb._effcal_model == 'vidmar-7' and cb._summing is not None
    assert cb.eff(121.7817)/off.eff(121.7817) > 1.02


@requires_data('decay')
@requires_pairs()
def test_summing_no_out_of_range_warning(eu_fitted, pairs_usable, capsys):
    # the factors take the curve at partner energies far below the fitted
    # range without the extrapolation warning
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys)
    assert cb._summing is not None
    assert not [l for l in log if 'WARNING' in l]
    assert 'outside the fitted range' not in capsys.readouterr().out


@requires_data('decay')
@requires_pairs()
def test_summing_entry_follows_the_parameters(eu_fitted, pairs_usable):
    # the entry belongs to the parameters fitted with it: assigning effcal
    # clears it, and a later calibrate() with summing=False leaves none
    cb, _ = _calibrate([eu_fitted], EU_SOURCES)
    assert cb._summing['L'] == cal_mod._SUMMING_L
    cb.effcal = cb.effcal
    assert cb._summing is None
    cb.calibrate([copy.deepcopy(eu_fitted)], sources=EU_SOURCES)
    assert cb._summing is not None
    cb.calibrate([copy.deepcopy(eu_fitted)], sources=EU_SOURCES, summing=False)
    assert cb._summing is None and cb.fit_config['summing'] is False
