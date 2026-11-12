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
import json
import os
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


@pytest.fixture
def shipped_stamps(datadir):
    """The isolated data directory with the release URLs of the shipped
    registry, so the expected stamps are those of the shipped files."""
    with open(os.path.join(os.path.dirname(data.__file__), 'data_registry.json')) as f:
        shipped = json.load(f)
    datadir['registry']['base_url'] = shipped['base_url']
    datadir['registry']['revisions'] = shipped.get('revisions', {})
    return datadir


STALE_DECAY = 'decay.db is from data generation v2, but this curie release uses generation v2.1.'


@pytest.mark.parametrize('stamp, usable', [('v2', False), ('v2.1', True)])
def test_pair_table_usable_by_the_shipped_stamps(shipped_stamps, capsys, stamp, usable):
    # the decay.db of the generation release is stamped v2: under the shipped
    # registry its pair table may not be used, and the revised file's may
    _decay_db(shipped_stamps['cache'] / 'decay.db', stamp)
    assert data._pair_table_usable() is usable
    out = capsys.readouterr().out
    if usable:
        assert 'generation' not in out
    else:
        assert STALE_DECAY in out and "ci.download('decay')" in out


# ---------------------------------------------------------------------------
# calibrate(): the summing correction of the efficiency fit

EU_SOURCES = [{'isotope': '152EU', 'A0': 3.7E4, 'ref_date': '01/01/2009 12:00:00'}]
E_CURVE = np.array([121.7817, 344.2785, 778.9045, 1408.013])

# The example spectrum calibrated by the parent of the summing correction with
# default settings, recorded 2026-11-10 at commit 1040f36: the curve and its
# fitted uncertainty at E_CURVE, the model tag and the calibrate() log lines.
# The comparisons against these values carry rtol 1E-3 for the curve and 1E-2
# for its uncertainty. Across OpenBLAS kernel types (SkylakeX, Haswell, Nehalem,
# Prescott, Zen, ...) and numpy 1.26 and 2.2 the largest relative difference
# measured on 2026-11-11 is 2.8E-5 in the curve and 7.6E-4 in its uncertainty,
# both from the fit's stopping point; the correction moves the curve by 4 to 6%
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


def _records(sp, app, lines):
    """Per line label, the record of the lines fit_peaks merged into its peak, or None."""
    return [sp._merged.get((app[ln][0], app[ln][1])) for ln in lines]


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
    np.testing.assert_allclose(cb.eff(E_CURVE), PARENT_EFF, rtol=1E-3)
    np.testing.assert_allclose(cb.unc_eff(E_CURVE), PARENT_UNC_EFF, rtol=1E-2)
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
    F, c, _ = cb._summing_factors([app[ln][0] for ln in lines], [app[ln][1] for ln in lines],
                                  cb.effcal, cb._summing['xray_partners'], merged=_records(eu_fitted, app, lines))
    off, _ = _calibrate([eu_fitted], EU_SOURCES, outlier_sigma=1.5, summing=False)
    off_pts = {ln: v for ln, v in _stored(off).items()}
    assert np.all(c > 0)
    for ln, f, cc in zip(lines, F, c):
        assert f != 1.0
        assert pts[ln][0] == pytest.approx(app[ln][2]/f, rel=1E-12)
        # the relative variance is the measured one plus the factor's term,
        # (k c)^2 with k = 0.3, as the docstring states, and c at the final curve
        assert (pts[ln][1]/pts[ln][0])**2 == pytest.approx(
            (off_pts[ln][1]/off_pts[ln][0])**2+(0.3*cc)**2, rel=1E-12)
    # c is (1 - loss) + gain, not |F - 1|: the two differ by twice the gain
    assert np.max(np.abs(c-np.abs(F-1.0))) > 1E-4


@requires_data('decay')
@requires_pairs()
def test_summing_outlier_screen_uses_the_stored_uncertainty(eu_fitted, pairs_usable):
    # the outlier screen judges a point by its own quoted uncertainty, which
    # for a corrected point holds the factor's term: a point is flagged exactly
    # when its stored residual from the curve exceeds outlier_sigma
    clip = 1.0
    cb, _ = _calibrate([eu_fitted], EU_SOURCES, outlier_sigma=clip)
    kept, drop = cb._calib_data['effcal'], cb._calib_data['effcal_dropped']
    out = np.array([str(r).startswith('outlier') for r in drop['reason']])
    z_out = (drop['efficiency'][out]-cb.eff(drop['energy'][out], cb.effcal))/drop['unc_efficiency'][out]
    z_kept = (kept['efficiency']-cb.eff(kept['energy'], cb.effcal))/kept['unc_efficiency']
    assert out.sum() > 0
    assert np.all(np.abs(z_out) >= clip) and np.all(np.abs(z_kept) < clip)


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


# the nuclides of the example spectrum and their daughters
EU_DECAY_NUCLIDES = ('152EU', '152SM', '152GD', '40K', '40CA', '40AR')


@pytest.fixture(scope='module')
def eu_decay_rows():
    """The schema of the installed decay database and its rows of
    EU_DECAY_NUCLIDES (all of the elements table), read once before any
    test redirects the data directory."""
    con = data._get_connection('decay')
    out = {}
    for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall():
        cols = [r[1] for r in con.execute('PRAGMA table_info("{}")'.format(name))]
        if name == 'elements':
            rows = con.execute('SELECT * FROM elements').fetchall()
        elif 'isotope' in cols and name != '_version':
            rows = con.execute('SELECT * FROM "{0}" WHERE isotope IN ({1})'.format(
                name, ','.join('?'*len(EU_DECAY_NUCLIDES))), EU_DECAY_NUCLIDES).fetchall()
        else:
            rows = []
        out[name] = (sql, len(cols), rows)
    return out


def _small_decay_db(path, tables, generation):
    """A decay database of the given rows, stamped `generation`."""
    con = sqlite3.connect(str(path))
    for name, (sql, ncol, rows) in tables.items():
        con.execute(sql)
        if name == '_version':
            rows = [('decay', generation)+(None,)*(ncol-2)]
        con.executemany('INSERT INTO "{0}" VALUES ({1})'.format(name, ','.join('?'*ncol)), rows)
    con.commit()
    con.close()


@requires_data('decay')
@requires_pairs()
@pytest.mark.parametrize('stamp', ['v2', 'v2.1'])
def test_summing_follows_the_shipped_stamps(eu_fitted, eu_decay_rows, shipped_stamps, capsys, stamp):
    # a decay database stamped as the generation release's decay.db gives the
    # generation warning and the result of summing=False with its message; one
    # stamped as the revised file gives the correction without either
    _small_decay_db(shipped_stamps['cache'] / 'decay.db', eu_decay_rows, stamp)
    capsys.readouterr()
    off, _ = _calibrate([eu_fitted], EU_SOURCES, summing=False)
    first = capsys.readouterr().out
    cb, log = _calibrate([eu_fitted], EU_SOURCES, capsys)
    no_correction = [l for l in log if 'applies no coincidence-summing correction' in l]
    if stamp == 'v2':
        assert STALE_DECAY in first
        _same_result(cb, off)
        assert cb._summing is None
        assert no_correction == ["[WARNING] Calibration.calibrate: calibrate() applies no coincidence-summing correction, "
                                 "because the decay database failed the data-generation check or has no coincidences "
                                 "table. Run ci.download('decay') to update it."]
    else:
        assert 'generation' not in first
        assert no_correction == []
        assert cb._summing is not None
        assert not np.allclose(cb.effcal, off.effcal, rtol=1E-3)


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


# ---------------------------------------------------------------------------
# the uncertainty term of the correction, and peaks of merged lines

@requires_data('decay')
@requires_pairs()
def test_summing_term_needs_pair_data(pairs_usable):
    # 137Cs has no coincidence pairs, and the pairs of 139Ce 165.86 keV are
    # its K x-rays, which a set without 152Eu leaves out (scale 0): each point
    # keeps factor 1 and the uncertainty of summing=False; the 60Co points
    # carry the term
    nuc = ['133BA', '137CS', '139CE', '60CO']
    sp = _synthetic(nuc, 0.0)
    cb, _ = _calibrate([sp], _sources(nuc))
    off, _ = _calibrate([sp], _sources(nuc), summing=False)
    assert cb._summing['xray_partners'] == 0.0
    on_pts, off_pts = _stored(cb), _stored(off)
    for ln in ('137CS:661.66', '139CE:165.86'):
        assert on_pts[ln][:2] == off_pts[ln][:2]
    F, c, _ = cb._summing_factors(['137CS', '139CE', '60CO', '60CO'], [661.657, 165.8575, 1173.228, 1332.492],
                                  cb.effcal, 0.0)
    assert list(F[:2]) == [1.0, 1.0] and list(c[:2]) == [0.0, 0.0] and np.all(c[2:] > 0)
    # 165.86 keV has pairs: with the x-ray partners in, it takes a correction
    assert cb._summing_factors(['139CE'], [165.8575], cb.effcal, 1.0)[1][0] > 0
    for ln, cc in zip(('60CO:1173.23', '60CO:1332.49'), c[2:]):
        assert (on_pts[ln][1]/on_pts[ln][0])**2 == pytest.approx(
            (off_pts[ln][1]/off_pts[ln][0])**2+(cal_mod._SUMMING_K*cc)**2, rel=1E-12)


@requires_data('decay')
@requires_pairs()
def test_summing_term_joins_the_line_group(eu_fitted, pairs_usable, monkeypatch):
    # two copies of the example spectrum give two points of each line.  The
    # pass with independent uncertainties takes the uncertainties of
    # summing=False; in the correlated pass the term (k c)^2, with c at the
    # curve of the independent pass, is common to the two points of a line
    # and absent between points of different lines
    calls = []
    real = cal_mod.curve_fit

    def spy(f, xdata, ydata, *a, **kw):
        out = real(f, xdata, ydata, *a, **kw)
        if kw.get('absolute_sigma'):
            calls.append((f, np.array(xdata), np.array(ydata), np.array(kw['sigma']), out[0]))
        return out
    monkeypatch.setattr(cal_mod, 'curve_fit', spy)

    def passes(**kw):
        calls.clear()
        cb, _ = _calibrate([eu_fitted, eu_fitted], EU_SOURCES, **kw)
        (f, x, y, s1, f0), (_, _, _, V, _) = calls[0], calls[1]
        assert s1.ndim == 1 and V.ndim == 2
        m0 = f(x, *f0)
        return cb, x, y, s1, f0, V/np.outer(m0, m0)
    cb, x, y, s1, f0, R = passes()
    _, x_off, _, s1_off, _, R_off = passes(summing=False)
    np.testing.assert_array_equal(x, x_off)
    K = cal_mod._SUMMING_K
    np.testing.assert_array_equal(s1, s1_off)

    app = _apparent(eu_fitted, EU_SOURCES)
    rec = [eu_fitted._merged.get(('152EU', e)) for e in x]
    c0 = cb._summing_factors(['152EU']*len(x), x, f0, cb._summing['xray_partners'], merged=rec)[1]
    n = len(x)//2
    assert np.all(x[:n] == x[n:]) and len(set(x[:n])) == n and len(app) > n
    for i in range(n):
        j, l = i+n, (i+1) % n
        # within a line, beyond what the line intensity and the source give
        within = (R[i, j]-R[i, l])-(R_off[i, j]-R_off[i, l])
        assert within == pytest.approx((K*c0[i])**2, rel=1E-6)
        # between lines, the source alone, as with summing=False
        assert R[i, l] == pytest.approx(R_off[i, l], rel=1E-9)


# the peak table of the example spectrum, 152EU and 40K, as fitted by curie
# before the merge record (2026-11-10, the commit that put the summing factor
# into the efficiency fit): counts and their uncertainties, in table order
PARENT_COUNTS = [498785.155, 80193.1932, 609.972482, 852.788649, 4257.9143, 641.658994, 1054.88421, 213989.691,
                 7079.93714, 14660.7423, 573.302851, 18491.7567, 2192.28175, 728.681862, 2391.43023, 652.599221,
                 2043.76723, 778.394122, 629.271838, 1775.75398, 3486.22644, 1283.07404, 721.849755, 46800.2794,
                 1196.60958, 532.129134, 13242.2312, 1356.8309, 783.181916, 4840.48565, 37803.1204, 2714.87597,
                 662.120846, 27664.5469, 4587.13596, 900.967652, 34946.2677, 3335.3372, 551.46015, 209.268764,
                 3666.64726, 43377.839, 1270.52482, 2985.81214, 1228.39969]
PARENT_UNC_COUNTS = [3246.89023, 398.271604, 110.530017, 92.5325342, 97.8909421, 64.4358652, 78.3026636, 631.548355,
                     248.883122, 131.788496, 65.4636603, 144.284397, 77.3441611, 60.2517687, 77.7755754, 58.5903803,
                     154.85147, 81.1213757, 60.5597179, 77.68501, 81.6901272, 87.8282054, 73.9118728, 296.163184,
                     70.1213505, 66.7081436, 137.53278, 75.2461525, 66.4338898, 3080.61654, 3094.0006, 91.4298982,
                     99.1481499, 211.219953, 91.0027416, 57.9582325, 211.686238, 75.1024858, 41.167192, 29.5011023,
                     68.9594214, 573.185972, 112.744644, 128.824277, 70.4921854]


@requires_data('decay')
def test_merge_record_changes_no_fit(eu_fitted):
    # fit_peaks records the lines it merges, each with its tabulated
    # intensity: 152Eu 443.96 + 444.01 keV and 719.35 + 719.36 keV are each one
    # peak with the summed intensity, and the record leaves the peak table as
    # it was. The counts agree to 4E-8 across OpenBLAS kernel types and numpy
    # versions (2026-11-11) and are compared at 1E-5. Their uncertainties come
    # from the fit's covariance at its stopping point and differ by up to 1.13E-3
    # (344.28 keV), so they are compared at 1E-2: a margin of 8.8 over the
    # measured difference; a change of the fit that moves them by 1.1% is rejected
    g = ci.Isotope('152EU').gammas()
    I = {e: float(g.loc[g['energy'] == e, 'intensity'].iloc[0])*1E-2 for e in (443.9606, 444.01, 719.346, 719.36)}
    assert eu_fitted._merged == {('152EU', 443.9606): ((443.9606, 444.01), (I[443.9606], I[444.01])),
                                 ('152EU', 719.346): ((719.346, 719.36), (I[719.346], I[719.36]))}
    pk = eu_fitted.peaks
    for (iso, e), (es, Is) in eu_fitted._merged.items():
        row = pk[(pk['isotope'] == iso) & (pk['energy'] == e)]
        assert len(row) == 1 and row.iloc[0]['intensity'] == sum(Is)
        assert es[1] not in pk['energy'].to_numpy()
    np.testing.assert_allclose(pk['counts'], PARENT_COUNTS, rtol=1E-5)
    np.testing.assert_allclose(pk['unc_counts'], PARENT_UNC_COUNTS, rtol=1E-2)


@requires_data('decay')
@requires_pairs()
def test_summing_merged_peak_takes_weighted_factor(eu_fitted, pairs_usable):
    # the point of the merged 443.96 keV peak is its measured value over the
    # intensity-weighted mean of the factors of 443.96 and 444.01 keV, not over
    # the factor of 443.96 keV alone
    cb, _ = _calibrate([eu_fitted], EU_SOURCES)
    (e1, e2), (i1, i2) = eu_fitted._merged[('152EU', 443.9606)]
    F, c, _ = cb._summing_factors(['152EU', '152EU'], [e1, e2], cb.effcal, cb._summing['xray_partners'])
    Fw = (i1*F[0]+i2*F[1])/(i1+i2)
    meas = _apparent(eu_fitted, EU_SOURCES)['152EU:443.96'][2]
    pt = _stored(cb)['152EU:443.96']
    assert pt[0] == pytest.approx(meas/Fw, rel=1E-12)
    assert abs(pt[0]/(meas/F[0])-1.0) > 1E-3
    # and its c is the weighted mean too
    Fm, cm, _ = cb._summing_factors(['152EU'], [e1], cb.effcal, cb._summing['xray_partners'],
                                    merged=[((e1, e2), (i1, i2))])
    assert Fm[0] == pytest.approx(Fw, rel=1E-14) and cm[0] == pytest.approx((i1*c[0]+i2*c[1])/(i1+i2), rel=1E-14)
    assert cm[0] > 0


# ---------------------------------------------------------------------------
# the calibration file: the summing entry

# a calibration file as curie writes it without the summing correction, and
# its curve and fitted uncertainty at E_CURVE as the parent of the stored
# entry evaluates them (recorded 2026-11-10 at commit 6dbe2a0). The file is
# evaluated, not fitted. Across OpenBLAS kernel types and numpy 1.26 and 2.2 the
# largest relative difference measured on 2026-11-11 is 1.0E-15 in the curve and
# 1.7E-14 in its uncertainty (the evaluation's last digit; 2.9E-12 for the
# uncertainty of another stored curve); the curve is compared at rtol 1E-12 and
# the uncertainty at 1E-10
PARENT_FILE = '''{
    "engcal": [0.0, 0.25, 0.0],
    "effcal": [0.045, 6.0, 2.0, 1.8, 0.4],
    "unc_effcal": [[1e-06, 0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0025, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0016, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0004]],
    "rescal": [1.87, 0.00027],
    "effcal_model": "vidmar-5",
    "effcal_erange": [121.7817, 1408.013]
}'''
PARENT_FILE_EFF = [0.03189717972421109, 0.01329977564289112, 0.005644427194924109, 0.0029125143038606113]
PARENT_FILE_UNC_EFF = [0.0007657763551727291, 0.0005982847235365494, 0.00034299514183142385, 0.00020393653651147277]


def _file(tmp_path, entry=None, name='cal.json'):
    js = json.loads(PARENT_FILE)
    if entry is not None:
        js['summing'] = entry
    fn = str(tmp_path / name)
    with open(fn, 'w') as f:
        json.dump(js, f)
    return fn


def test_summing_file_of_the_parent(tmp_path):
    # a file without the entry loads as an apparent curve and evaluates as the
    # parent evaluates it; saved again it writes no entry
    fn = str(tmp_path / 'parent.json')
    with open(fn, 'w') as f:
        f.write(PARENT_FILE)
    cb = ci.Calibration(fn)
    assert cb._summing is None and cb._effcal_model == 'vidmar-5'
    np.testing.assert_allclose(cb.eff(E_CURVE), PARENT_FILE_EFF, rtol=1E-12)
    np.testing.assert_allclose(cb.unc_eff(E_CURVE), PARENT_FILE_UNC_EFF, rtol=1E-10)
    cb.saveas(str(tmp_path / 'again.json'))
    with open(str(tmp_path / 'again.json')) as f:
        assert json.load(f) == json.loads(PARENT_FILE)


def test_summing_file_entry_leaves_the_curve(tmp_path):
    # the entry is read back, changes no evaluation of the curve, and an item
    # of it that curie does not know is ignored (and not written again)
    cb = ci.Calibration(_file(tmp_path, {'L': 0.69, 'xray_partners': 0.58, 'note': 'other'}))
    assert cb._summing == {'L': 0.69, 'xray_partners': 0.58}
    np.testing.assert_allclose(cb.eff(E_CURVE), PARENT_FILE_EFF, rtol=1E-12)
    np.testing.assert_allclose(cb.unc_eff(E_CURVE), PARENT_FILE_UNC_EFF, rtol=1E-10)
    cb.saveas(str(tmp_path / 'again.json'))
    with open(str(tmp_path / 'again.json')) as f:
        assert json.load(f)['summing'] == {'L': 0.69, 'xray_partners': 0.58}


@requires_data('decay')
@requires_pairs()
def test_summing_file_round_trip(eu_fitted, pairs_usable, tmp_path):
    # a corrected curve writes the entry, with L and the scale, and a file read
    # back holds them; a curve fitted with summing=False writes none
    cb, _ = _calibrate([eu_fitted], EU_SOURCES)
    fn = str(tmp_path / 'on.json')
    cb.saveas(fn)
    with open(fn) as f:
        assert json.load(f)['summing'] == {'L': cal_mod._SUMMING_L, 'xray_partners': cb._summing['xray_partners']}
    back = ci.Calibration(fn)
    assert back._summing == cb._summing and back._summing['xray_partners'] > 0
    np.testing.assert_array_equal(back.eff(E_CURVE), cb.eff(E_CURVE))
    off, _ = _calibrate([eu_fitted], EU_SOURCES, summing=False)
    off.saveas(str(tmp_path / 'off.json'))
    with open(str(tmp_path / 'off.json')) as f:
        assert 'summing' not in json.load(f)
    assert ci.Calibration(str(tmp_path / 'off.json'))._summing is None


@requires_data('decay')
@requires_pairs()
def test_summing_file_entry_ignored_under_loglog(eu_fitted, pairs_usable, tmp_path, capsys):
    # the correction needs a Vidmar curve: an entry in a file whose curve is
    # loglog is not read back, one warning says so, and the peaks of a
    # spectrum on that calibration take the curve value
    cb, _ = _calibrate([eu_fitted], EU_SOURCES, summing=False, effcal_model='loglog')
    fn = _file(tmp_path, name='loglog.json')
    cb.saveas(fn)
    with open(fn) as f:
        js = json.load(f)
    assert js['effcal_model'] == 'loglog-4'
    js['summing'] = {'L': 0.69, 'xray_partners': 0.58}
    with open(fn, 'w') as f:
        json.dump(js, f)
    capsys.readouterr()
    back = ci.Calibration(fn)
    log = [l for l in capsys.readouterr().out.splitlines() if 'summing' in l]
    assert back._summing is None
    assert log == ['[WARNING] Calibration: curie ignores the coincidence-summing entry of {0}. The correction needs '
                   'a Vidmar efficiency model, and the file holds a loglog-4 curve.'.format(fn)]
    sp, _ = _refit(back, sp0=eu_fitted)
    np.testing.assert_array_equal(sp.peaks['efficiency'], back.eff(sp.peaks['energy']))
    np.testing.assert_array_equal(sp.peaks['unc_efficiency'], back.unc_eff(sp.peaks['energy']))


@requires_data('decay')
@requires_pairs()
def test_summing_entry_carried_to_the_spectra(eu_fitted, pairs_usable):
    # the spectra a corrected calibration is carried to hold its entry; after
    # a second calibrate() with summing=False they hold none
    sp = copy.deepcopy(eu_fitted)
    cb = ci.Calibration()
    cb.calibrate([sp], sources=EU_SOURCES)
    assert cb._summing is not None and sp.cb._summing == cb._summing
    assert sp.cb._summing is not cb._summing
    cb.calibrate([sp], sources=EU_SOURCES, summing=False)
    assert cb._summing is None and sp.cb._summing is None


# ---------------------------------------------------------------------------
# Spectrum: the peak table of a spectrum whose calibration carries the entry

@pytest.fixture(scope='module')
def eu_corrected(eu_fitted):
    """The example spectrum's calibration fitted with the summing correction."""
    with contextlib.redirect_stdout(io.StringIO()):
        cb, _ = _calibrate([eu_fitted], EU_SOURCES)
    return cb


def _refit(cb, capsys=None, sp0=None, entry=True):
    """The example spectrum (or sp0) refitted on a copy of cb, with the entry or without
    it (the same curve); returns the spectrum and its log lines."""
    if capsys is not None:
        capsys.readouterr()
    sp = copy.deepcopy(sp0)
    sp.cb = cb
    if not entry:
        sp.cb._summing = None
    if capsys is None:
        with contextlib.redirect_stdout(io.StringIO()):
            sp.fit_peaks()
        return sp, []
    sp.fit_peaks()
    return sp, [l for l in capsys.readouterr().out.splitlines() if 'Spectrum' in l]


def _factors_of(sp):
    """The factor and c of each peak of sp at its calibration's entry."""
    pk = sp.peaks
    rec = [sp._merged.get((i, float(e))) for i, e in zip(pk['isotope'], pk['energy'])]
    F, c, _ = sp.cb._summing_factors(pk['isotope'].tolist(), pk['energy'].to_numpy(), sp.cb.effcal,
                                     sp.cb._summing['xray_partners'], L=sp.cb._summing['L'], merged=rec)
    return F, c


@requires_data('decay')
def test_peaks_without_entry_are_the_parent(eu_fitted, monkeypatch):
    # without the entry no factor, scale or table check is reached, and each
    # efficiency column is the curve: the default calibration, and one fitted
    # with summing=False
    def unreachable(*a, **k):
        raise AssertionError('summing code reached without the entry')
    off, _ = _calibrate([eu_fitted], EU_SOURCES, summing=False)
    monkeypatch.setattr(cal_mod.Calibration, '_summing_factors', unreachable)
    monkeypatch.setattr(ci.spectrum, '_pair_table_usable', unreachable)
    for cb in (eu_fitted.cb, off):
        sp, _ = _refit(cb, sp0=eu_fitted)
        pk = sp.peaks
        assert sp.cb._summing is None
        np.testing.assert_array_equal(pk['efficiency'], sp.cb.eff(pk['energy']))
        np.testing.assert_array_equal(pk['unc_efficiency'], sp.cb.unc_eff(pk['energy']))
    np.testing.assert_allclose(_refit(eu_fitted.cb, sp0=eu_fitted)[0].peaks['counts'], PARENT_COUNTS, rtol=1E-5)


@requires_data('decay')
@requires_pairs()
def test_peaks_with_entry_take_the_factor(eu_fitted, eu_corrected, pairs_usable):
    # with the entry, the decays of each line are those of the same curve
    # without it over the line's factor; efficiency is the curve times the
    # factor, unc_efficiency adds the factor's term in quadrature, and the
    # decays follow from the table's own columns
    on_sp, _ = _refit(eu_corrected, sp0=eu_fitted)
    off_sp, _ = _refit(eu_corrected, sp0=eu_fitted, entry=False)
    a, b = on_sp.peaks, off_sp.peaks
    np.testing.assert_array_equal(a['counts'], b['counts'])
    F, c = _factors_of(on_sp)
    eu = (a['isotope'] == '152EU').to_numpy()
    assert eu.sum() > 30 and np.all(c[eu] > 0) and np.all(np.abs(F[eu]-1.0) > 1E-4)
    np.testing.assert_allclose(a['decays'], b['decays']/F, rtol=1E-12)
    np.testing.assert_allclose(a['efficiency'], b['efficiency']*F, rtol=1E-14)
    rel2 = (b['unc_efficiency']/b['efficiency'])**2+(cal_mod._SUMMING_K*c)**2
    np.testing.assert_allclose(a['unc_efficiency']/a['efficiency'], np.sqrt(rel2), rtol=1E-12)
    lt = on_sp.live_time/on_sp.real_time
    np.testing.assert_allclose(a['decays'], a['counts']/(a['intensity']*a['efficiency']*lt), rtol=1E-12)
    np.testing.assert_allclose(a['unc_decays']/a['decays'], np.sqrt((a['unc_counts']/a['counts'])**2+(
        a['unc_efficiency']/a['efficiency'])**2+(a['unc_intensity']/a['intensity'])**2), rtol=1E-12)
    # the merged 443.96 keV peak takes the weighted factor of its lines
    e1, e2 = on_sp._merged[('152EU', 443.9606)][0]
    Fm = on_sp.cb._summing_factors(['152EU'], [e1], on_sp.cb.effcal, on_sp.cb._summing['xray_partners'],
                                   merged=[on_sp._merged[('152EU', 443.9606)]])[0][0]
    F1 = on_sp.cb._summing_factors(['152EU'], [e1], on_sp.cb.effcal, on_sp.cb._summing['xray_partners'])[0][0]
    i = int(np.where(a['energy'] == e1)[0][0])
    assert F[i] == Fm and abs(Fm/F1-1.0) > 1E-3


@requires_data('decay')
@requires_pairs()
def test_peaks_with_entry_in_decay_chain(eu_fitted, eu_corrected, pairs_usable, tmp_path):
    # get_counts decomposes unc_counts into its three classes (counting, line
    # intensity, efficiency), which add in quadrature to it; a saved peak file
    # read back gives the same table and the same counts, to the precision of
    # the file (the json writer keeps 10 decimal places)
    sp, _ = _refit(eu_corrected, sp0=eu_fitted)
    eob = '01/01/2009 12:00:00'
    dc = ci.DecayChain('152EU', A0=3.7E4, units='h')
    with contextlib.redirect_stdout(io.StringIO()):
        dc.get_counts([sp], EoB=eob)
    ct = dc.counts
    np.testing.assert_allclose(np.sqrt(ct['unc_stat']**2+ct['unc_line']**2+ct['unc_corr']**2), ct['unc_counts'], rtol=1E-12)
    pk = sp.peaks[sp.peaks['isotope'] == '152EU']
    np.testing.assert_allclose(np.sort(ct['unc_corr']/ct['counts']), np.sort(pk['unc_efficiency']/pk['efficiency']), rtol=1E-12)
    cols = ['energy', 'counts', 'efficiency', 'unc_efficiency', 'decays', 'unc_decays']
    for ext, rtol in (('csv', 1E-12), ('json', 1E-5)):
        fn = str(tmp_path / ('peaks.'+ext))
        sp.saveas(fn)
        back = pd.read_csv(fn) if ext == 'csv' else pd.read_json(fn, orient='records')
        np.testing.assert_allclose(back[cols].to_numpy(dtype=float), sp.peaks[cols].to_numpy(dtype=float), rtol=rtol)
        dc_f = ci.DecayChain('152EU', A0=3.7E4, units='h')
        with contextlib.redirect_stdout(io.StringIO()):
            dc_f.get_counts([sp.filename], EoB=eob, peak_data=fn)
        for col in ('counts', 'unc_counts', 'unc_corr'):
            np.testing.assert_allclose(dc_f.counts[col], ct[col], rtol=rtol)


@requires_data('decay')
@requires_pairs()
def test_peaks_stale_table(eu_fitted, eu_corrected, capsys, monkeypatch):
    # a pair table that may not be used: the table of the curve without the
    # entry, and one message
    off_sp, _ = _refit(eu_corrected, sp0=eu_fitted, entry=False)
    monkeypatch.setitem(data._get_connection('decay').cache, 'pair_table_usable', False)
    sp, log = _refit(eu_corrected, capsys, sp0=eu_fitted)
    pd.testing.assert_frame_equal(sp.peaks, off_sp.peaks, check_exact=True)
    msg = [l for l in log if 'summing' in l]
    assert msg == ["[WARNING] Spectrum(eu_calib_7cm.Spe).fit_peaks: the peak efficiencies take no coincidence-summing "
                   "correction, because the decay database failed the data-generation check or has no coincidences "
                   "table. Run ci.download('decay') to update it."]


@requires_data('decay')
@requires_pairs()
def test_peaks_guarded_line_and_no_range_warning(eu_fitted, eu_corrected, capsys, monkeypatch):
    # the factors raise no warning of their own (the curve's own range
    # warnings are those of the table without the entry); then, with pair rows
    # in which a partner of 344.28 keV is three times as intense as the line,
    # that line keeps factor 1 and one warning names it
    _, log_off = _refit(eu_corrected, capsys, sp0=eu_fitted, entry=False)
    sp, log = _refit(eu_corrected, capsys, sp0=eu_fitted)
    assert [l for l in log if 'WARNING' in l] == [l for l in log_off if 'WARNING' in l]
    off = _refit(eu_corrected, sp0=eu_fitted, entry=False)[0].peaks
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
    con = data._get_connection('decay')
    monkeypatch.setattr(con, 'cache', {k: v for k, v in con.cache.items() if k in ('generation_ok', 'pair_table_usable')})
    sp, log = _refit(eu_corrected, capsys, sp0=eu_fitted)
    pk = sp.peaks
    assert [l for l in log if 'no coincidence-summing correction for' in l] == [
        '[WARNING] Spectrum(eu_calib_7cm.Spe).fit_peaks: no coincidence-summing correction for 152EU 344.3 keV: in '
        'the decay data a coincidence pair of each of these lines is more intense than the line itself.']
    i, j = [int(np.where(pk['energy'] == e)[0][0]) for e in (344.2785, 778.9045)]
    assert pk['efficiency'][i] == off['efficiency'][i] and pk['unc_efficiency'][i] == off['unc_efficiency'][i]
    assert abs(pk['efficiency'][j]/off['efficiency'][j]-1.0) > 1E-3


@requires_data('decay')
@requires_pairs()
def test_peaks_nuclide_outside_the_calibration_set(eu_synthetic, pairs_usable):
    # 60Co is not in the calibration set: its lines take their factors from
    # the stored entry and the curve
    cb, _ = _calibrate([eu_synthetic], _sources(['152EU']))
    co = _synthetic(['60CO'], SYN_SCALE)
    sp, _ = _refit(cb, sp0=co)
    off, _ = _refit(cb, sp0=co, entry=False)
    F, c = _factors_of(sp)
    assert np.all(c > 0) and np.all(F < 0.99) and len(F) == 2
    np.testing.assert_allclose(sp.peaks['decays'], off.peaks['decays']/F, rtol=1E-12)
    np.testing.assert_allclose(sp.peaks['efficiency'], cb.eff(sp.peaks['energy'])*F, rtol=1E-14)
