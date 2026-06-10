"""Library cross-section reference tests (public suite, category 4).

Every value here is pinned to the *shipped* nuclear-data libraries, recorded 2026-06-09 on
curie v0.0.34: IRDFF-II (IRDFF.db), ENDF/B-VII.1 (endf.db), IAEA charged-particle monitors
(iaea_monitors.db, ~2017 baseline). They are EXPECTED to change at the Stage-6 data
rebuild - that is what the `library_value` marker means: re-record each value deliberately
against the new evaluation, per the re-baseline checklist. An unexpected shift at any
other time is a regression in retrieval/interpolation/quadrature, not in data.

Tolerances are tight (1E-6): these are deterministic database reads followed by linear
interpolation and quadrature.

Independent context for the Stage-6 reviewer (deliberately NOT asserted - evaluations
fluctuate): thermal 59Co(n,g) = 37.18 b is the classic activation standard, and thermal
115In(n,g)116mIn ~ 162 b; the pinned values agree within evaluation differences.

Operations covered per reaction: interpolate, integrate, and flux-spectrum average - the
last two exercise the midpoint quadrature introduced in v0.0.34 (commit c36d364).
"""
import numpy as np
import pytest

import curie as ci
from conftest import requires_data

pytestmark = pytest.mark.library_value

REL = 1E-6


@requires_data('IRDFF')
class TestIn115ng:
    """115IN(n,g)116INm, IRDFF-II. Neutron dosimetry standard."""

    @pytest.fixture(scope='class')
    def rx(self):
        return ci.Reaction('115IN(n,g)', 'irdff')

    def test_name_resolution(self, rx):
        assert rx.name == '115IN(n,g)116INm'

    def test_interpolate(self, rx):
        assert float(rx.interpolate(2.53E-8)) == pytest.approx(159790.0, rel=REL)  # thermal
        assert float(rx.interpolate(0.5)) == pytest.approx(161.41646650941306, rel=REL)

    def test_integrate(self, rx):
        assert float(rx.integrate(0.5, 2.0)) == pytest.approx(322.8329330188261, rel=REL)

    def test_average(self, rx):
        flux_E = np.linspace(0.5, 2.0, 151)
        avg = float(rx.average(flux_E, np.ones(151)))
        assert avg == pytest.approx(164.98104222330275, rel=REL)


@requires_data('endf')
class TestCo59ng:
    """59CO(n,g)60CO, ENDF/B-VII.1. Classic activation standard."""

    @pytest.fixture(scope='class')
    def rx(self):
        return ci.Reaction('59CO(n,g)', 'endf')

    def test_name_resolution(self, rx):
        assert rx.name == '59CO(n,g)60CO'

    def test_interpolate_thermal(self, rx):
        assert float(rx.interpolate(2.53E-8)) == pytest.approx(37173.5, rel=REL)

    def test_integrate(self, rx):
        assert float(rx.integrate(0.5, 2.0)) == pytest.approx(14.7, rel=REL)

    def test_average(self, rx):
        flux_E = np.linspace(0.5, 2.0, 151)
        avg = float(rx.average(flux_E, np.ones(151)))
        assert avg == pytest.approx(5.302980177664555, rel=REL)


@requires_data('iaea_monitors')
class TestNatCuP63Zn:
    """natCU(p,x)63ZN, IAEA charged-particle monitor.

    Chosen over natFE(p,x)51CR (maintainer decision 2026-06-09): the 63ZN energy grid is
    strictly increasing (verified, 4-100 MeV), so integrate/average are well-defined today;
    the natFE grid is one of the 9 unsorted groups and is covered as a known bug in the
    curie-validation regression tree until the Stage-1 sort-on-load fix.
    """

    @pytest.fixture(scope='class')
    def rx(self):
        return ci.Reaction('natCU(p,x)63ZN', 'iaea')

    def test_grid_sorted(self, rx):
        assert np.all(np.diff(rx.eng) > 0.0)

    def test_interpolate(self, rx):
        assert float(rx.interpolate(12.0)) == pytest.approx(343.15, rel=REL)
        assert float(rx.interpolate(25.0)) == pytest.approx(25.75, rel=REL)

    def test_interpolate_unc(self, rx):
        # library-provided uncertainty, not a fit uncertainty - safe to pin
        assert float(rx.interpolate_unc(12.0)) == pytest.approx(16.17, rel=REL)

    def test_integrate(self, rx):
        assert float(rx.integrate(10.0, 40.0)) == pytest.approx(11940.4, rel=REL)

    def test_average(self, rx):
        flux_E = np.linspace(10.0, 40.0, 31)
        avg = float(rx.average(flux_E, np.ones(31)))
        assert avg == pytest.approx(99.58161290322582, rel=REL)
