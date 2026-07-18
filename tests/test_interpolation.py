"""Reaction interpolation schemes and the interp_config surface.

The TENDL libraries default to 'pchip-sqrt' — monotone PCHIP interpolation in
sqrt(E)-sqrt(sigma) space, exact through the evaluated points with no
overshoot at thresholds (the previous quadratic spline oscillated across
sharp threshold rises) — and the pointwise-linearized libraries (ENDF via
RECONR, IRDFF's tabulated form, the IAEA evaluations) default to 'linear'.
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data


@requires_data('tendl_p_rp')
class TestPchipSqrt:

    @pytest.fixture()
    def rx(self):
        return ci.Reaction('natTi(p,x)48V', 'tendl_p')

    def test_tendl_defaults_to_pchip(self, rx):
        assert rx.interp_config == {'interpolation': 'pchip-sqrt'}

    def test_exact_through_nodes(self, rx):
        # every scheme must reproduce the evaluated points themselves
        mid = len(rx.eng) // 2
        assert float(rx.interpolate(rx.eng[mid])) == pytest.approx(rx.xs[mid], rel=1E-9)

    def test_no_overshoot(self, rx):
        e = np.linspace(rx.eng[0], rx.eng[-1], 20011)
        assert float(rx.interpolate(e).max()) <= float(rx.xs.max()) * (1 + 1E-9)
        assert float(rx.interpolate(e).min()) >= 0.0

    def test_zero_outside_grid_and_below_threshold(self, rx):
        assert float(rx.interpolate(rx.eng[-1] * 1.5)) == 0.0
        assert float(rx.interpolate(1E-3)) == 0.0

    def test_kwarg_updates_persist(self, rx):
        rx.interpolate(12.0, interpolation='linear')
        assert rx.interp_config == {'interpolation': 'linear'}

    def test_linear_override_matches_linear_interpolation(self, rx):
        e = np.linspace(rx.eng[0], rx.eng[-1], 501)
        got = rx.interpolate(e, interpolation='linear')
        want = np.maximum(np.interp(e, rx.eng, rx.xs), 0.0)
        np.testing.assert_allclose(got, want, rtol=1E-9)

    def test_unknown_key_warns_with_suggestion(self, rx, capsys):
        rx.interp_config = {'interploation': 'linear'}
        assert "did you mean 'interpolation'" in capsys.readouterr().out

    def test_invalid_scheme_raises(self, rx):
        with pytest.raises(ValueError, match="'interpolation' must be one of"):
            rx.interp_config = {'interpolation': 'spline'}


@requires_data('endf')
def test_pointwise_linearized_libraries_default_to_linear():
    # endf.db is RECONR-linearized to a 0.5% tolerance: lin-lin IS its
    # interpolation convention (IRDFF's tabulated form and the IAEA
    # evaluations likewise ship pointwise data intended for lin-lin)
    assert ci.Reaction('115IN(n,g)', 'endf').interp_config == {'interpolation': 'linear'}


@requires_data('iaea_monitors')
def test_iaea_defaults_to_linear():
    assert ci.Reaction('natCU(p,x)63ZN', 'iaea').interp_config == {'interpolation': 'linear'}
