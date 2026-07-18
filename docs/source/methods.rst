.. _methods:

====================
Theory & Methodology
====================

This chapter documents the models and numerical methods that Curie uses, in
enough detail to reproduce its results or to describe them in a publication.
The user's guide shows *how* to invoke these methods; this chapter defines
*what* is computed.

.. _methods_peak_fitting:

Gamma-ray Peak Fitting
----------------------

Peak shape
~~~~~~~~~~

Curie models a gamma-ray peak in a spectrum from a high-purity germanium
(HPGe) detector as a Gaussian with an optional low-energy skew component
and an optional step in the background, as a function of the
analog-to-digital converter (ADC) channel number :math:`x` (a bin index
that increases with detected energy, which the energy calibration below
maps to keV):

.. math::

   F(x) = A\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}
        + R\,A\,e^{\frac{x-\mu}{\alpha\sigma}}\,
          \mathrm{erfc}\!\left(\frac{x-\mu}{\sqrt{2}\sigma}
          + \frac{1}{\sqrt{2}\alpha}\right)
        + \mathrm{step}\cdot A\,
          \mathrm{erfc}\!\left(\frac{x-\mu}{\sqrt{2}\sigma}\right)

where :math:`A`, :math:`\mu` and :math:`\sigma` are the amplitude, centroid
and width of the Gaussian, and :math:`\mathrm{erfc}` is the complementary
error function.  The second term is a skewed-Gaussian tail on the
low-energy side of the peak, characteristic of incomplete charge collection;
its relative amplitude :math:`R` and decay constant :math:`\alpha` are shared
by all peaks in the spectrum.  The third term is a step function (amplitude
``step`` relative to the peak) that accounts for the difference in the
Compton continuum — the broad background left by gamma rays that deposited
only part of their energy in the detector — on either side of the peak.  By default :math:`R=0.1`,
:math:`\alpha=0.9` and :math:`\mathrm{step}=0`, and these three parameters
are held fixed; the ``skew_fit`` and ``step_fit`` options of
`Spectrum.fit_peaks()` add them to the fitted parameters instead.

Background
~~~~~~~~~~

The continuum under the peaks is modeled in one of two ways, selected by the
``bg`` option:

* A polynomial — constant, linear or quadratic in channel number — fit
  jointly with the peaks in each multiplet (a group of peaks close enough
  on the spectrum that their fit windows overlap and they must be fit
  together).

* The default, a variant of the SNIP algorithm of Ryan *et al.* [Ryan1988]_,
  which estimates a smooth background non-parametrically and removes it from
  the fit entirely.  The histogram is first compressed with the double-log
  operator :math:`v = \ln(\ln(\sqrt{y+1}+1)+1)`; each point is then
  iteratively replaced by the minimum of itself and the mean of its
  neighbors at :math:`\pm M w(x)`, where the half-width :math:`M w(x)`
  grows in ten steps to :math:`7.5` times the local peak width
  :math:`w(x)` from the resolution calibration.  Because the operator only
  ever lowers a point toward the average of its neighbors, structures
  narrower than the window (peaks) are clipped away while the smooth
  continuum is preserved.  The result is transformed back, given a small
  positive safety margin (about :math:`1.5\sqrt{B}`), and smoothed with a
  resolution-matched exponential filter.  The ``snip_adj`` option scales
  the window width and margin.

The SNIP background is appropriate when the continuum varies smoothly under
the peak; for peaks sitting on rapidly-varying features (e.g. the electron
backscatter edge), a polynomial background will perform better.

Peak selection
~~~~~~~~~~~~~~

The list of candidate peaks is generated from the gamma-ray lines of the
isotopes assigned to the spectrum (plus any user-supplied lines), filtered
by the criteria in ``fit_config``: minimum energy (``E_min``), minimum
intensity (``I_min``), proximity to the 511 keV annihilation line
(``dE_511``), and predicted signal-to-noise ratio.  The predicted SNR of a
line is

.. math::

   \mathrm{SNR} = \frac{A_{\mathrm{pred}}}{\sqrt{B_{\mathrm{SNIP}}(\mu)}}

where the predicted amplitude :math:`A_{\mathrm{pred}}` is computed from the
line intensity, the current efficiency calibration and the expected peak
width, and :math:`B_{\mathrm{SNIP}}(\mu)` is the SNIP background at the
expected centroid.  Lines with :math:`\mathrm{SNR} < \mathrm{SNR}_{\min}`
(default 4.0) are excluded.  Lines whose fit windows overlap are fit
together as a multiplet, up to ``multi_max`` peaks per fit; each fit spans
``pk_width`` (default 7.5) expected peak widths around the peaks.

Fitting and uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~

Each multiplet is fit by weighted least squares, minimizing

.. math::

   \chi^2 = \sum_i \frac{(y_i - F(x_i))^2}{y_i + 0.1}

i.e. with Poisson standard deviations as weights (the small constant keeps
empty channels finite).  Because these are true counting uncertainties, the
parameter covariance — the matrix of the fitted parameters' variances and
their correlations — is taken directly from the fit, without the customary
rescaling by the reduced chi-square :math:`\chi^2_\nu` (the :math:`\chi^2`
per degree of freedom, near one for a well-modeled fit) that would shrink
the uncertainties of clean peaks below the floor set by counting
statistics.  When the reduced
chi-square (evaluated over the non-empty channels) exceeds one, the
covariance is inflated by :math:`\chi^2_\nu`: an imperfect peak model can
increase the uncertainties, but never reduce them.

The net counts in a peak are the analytic integrals of the Gaussian and
skew terms of the fitted shape,

.. math::

   N_c = \sqrt{2\pi}\,A\sigma
       + 2\,R\,A\,\alpha\,\sigma\,e^{-\frac{1}{2\alpha^2}}

with uncertainty propagated from the full parameter covariance.  The step
term is part of the background and does not contribute counts.

The number of decays and the average decay rate during the count are

.. math::

   D = \frac{N_c}{I_\gamma\;\varepsilon(E)\;f_{\mathrm{corr}}\;
       (t_{\mathrm{live}}/t_{\mathrm{real}})}
   \qquad
   \bar{A} = \frac{D}{t_{\mathrm{real}}}

where :math:`I_\gamma` is the gamma intensity (branching ratio),
:math:`\varepsilon(E)` the peak efficiency, :math:`f_{\mathrm{corr}}` the
product of any geometry and attenuation corrections, and
:math:`t_{\mathrm{live}}/t_{\mathrm{real}}` the dead-time correction.  The
relative uncertainties of the counts, efficiency and intensity are combined
in quadrature; the counting statistics of the peak area are already
contained in the fitted covariance.

.. [Ryan1988] C.G. Ryan et al., "SNIP, a statistics-sensitive background
   treatment for the quantitative analysis of PIXE spectra in geoscience
   applications", *Nucl. Instrum. Methods Phys. Res. B* **34** (1988) 396.

.. _methods_calibration:

Detector Calibration
--------------------

`Calibration.calibrate()` performs three sequential fits to the peaks of
one or more spectra of reference sources with known activities: an energy
calibration, a resolution calibration, and an efficiency calibration.

Energy calibration
~~~~~~~~~~~~~~~~~~

The energy calibration maps ADC channel number to gamma-ray energy with a
linear, quadratic or cubic polynomial,

.. math::

   E(x) = a_0 + a_1 x \;(+\, a_2 x^2 + a_3 x^3)

fit by weighted least squares to the fitted peak centroids versus the known
line energies, weighted by the centroid uncertainties.  Points with energy
uncertainties larger than a set fraction of the energy
(``engcal_max_error``, default 25%) are excluded.  The model is selected
with ``fit_config['engcal_model']``; by default the form of the current
calibration is kept.  The linear and quadratic forms invert analytically
for the energy-to-channel map; the cubic is inverted numerically (the real
root nearest the linear estimate), and a fitted cubic whose derivative
changes sign inside the calibrated channel range is flagged as
non-monotonic with a warning.

Resolution calibration
~~~~~~~~~~~~~~~~~~~~~~

The peak width (the Gaussian :math:`\sigma`, in channels) is modeled as
one of three functions of channel number, selected with
``fit_config['rescal_model']``:

.. math::

   \sigma(x) = b_0 + b_1 x
   \qquad\text{or}\qquad
   \sigma(x) = b_0 \sqrt{x}
   \qquad\text{or}\qquad
   \sigma(x) = \sqrt{b_0 + b_1 x + b_2 x^2}

fit to the fitted peak widths.  The square-root form (``'sqrt'``) is the
expectation from pure counting statistics of charge-carrier generation; the
linear form (``'linear'``, the default) accounts for the additional
electronic-noise contribution typical of real HPGe systems.  The
square-root-of-quadratic form (``'sqrt_quad'``) is the modern
Genie/InterSpec-family model: its three terms under the root map onto the
physical width decomposition — constant electronic noise, charge-carrier
(Fano) statistics growing linearly, and charge-collection variations
growing quadratically — and it contains the ``'sqrt'`` form as the special
case :math:`b_0 = b_2 = 0`.  Strongly discrepant points (residuals beyond
``outlier_sigma``, default :math:`\sqrt{10} \approx 3.16` standard
deviations) are clipped from the retained calibration data; they remain
part of the fit itself and stay visible, with their reasons, in
``cb.rescal_data`` and the calibration plots.

Efficiency measurement
~~~~~~~~~~~~~~~~~~~~~~

Each fitted peak of a reference source provides one measurement of the
absolute peak efficiency at its energy.  Accounting for the decay of the
source during the count, the efficiency is

.. math::

   \varepsilon(E) = \frac{N_c\,\lambda}
       {f_{\mathrm{corr}}\,\left(1-e^{-\lambda t_{\mathrm{real}}}\right)
        e^{-\lambda t_d}\;I_\gamma\,A_0\,
        (t_{\mathrm{live}}/t_{\mathrm{real}})}

where :math:`A_0` is the reference activity, :math:`t_d` the elapsed time
from the reference date to the start of the count, :math:`\lambda` the
decay constant, and the remaining symbols are as defined above.

Efficiency model
~~~~~~~~~~~~~~~~

The efficiency curve is a modified form of the physically founded
efficiency model of Vidmar *et al.* [Vidmar2001]_, built from the
tabulated photon interaction coefficients of germanium — total attenuation
:math:`\mu(E)`, photoelectric :math:`\tau(E)` and Compton
:math:`\sigma_C(E)` — rather than from an arbitrary fitting function:

.. math::

   \varepsilon(E) = S\,\left(1-e^{-\mu(E) L}\right)\,
       \frac{\tau(E) + \sigma_C(E)\,
       \left(1-e^{-(\mu(E) L_0)^{\alpha}}\right)\kappa}{\mu(E)}

with five fitted parameters: a solid-angle scale :math:`S`, an effective
crystal length :math:`L`, and three parameters :math:`(L_0, \alpha,
\kappa)` describing the probability that a Compton-scattered photon is
subsequently absorbed.  The photoelectric term contributes full-energy
events directly; the Compton term contributes only when the scattered
photon is reabsorbed.  When x-rays are included in the peak fits
(``fit_config['xrays'] = True``), two low-energy attenuation factors extend
the model to seven parameters,

.. math::

   \varepsilon_7(E) = e^{-\mu_{\mathrm{w}}(E)\,w}\;
                      e^{-\mu(E)\,d}\;\varepsilon(E)

representing the detector entrance window (beryllium attenuation
coefficient :math:`\mu_{\mathrm{w}}`, thickness :math:`w`) and the
germanium dead layer (thickness :math:`d`).  Both the 5- and 7-parameter
forms are fit, and the model whose goodness-of-fit is closer to one is
kept (the choice is reported; ``effcal_model='vidmar-5'`` or
``'vidmar-7'`` forces one form).  The interaction coefficients are log-log
interpolations of the NIST XCOM photon cross-section tabulations.

With ``fit_config['effcal_model'] = 'loglog'`` the efficiency is instead
the standard empirical log-log polynomial,

.. math::

   \ln \varepsilon(E) = \sum_{i=0}^{n} a_i \,(\ln E)^i

with the order selected by the model name (``'loglog'`` is order 4;
``'loglog-2'`` through ``'loglog-8'`` choose it explicitly).
High-order log-log polynomials reproduce germanium efficiency curves very
well *within* the fitted energy range [Kis1998]_, but being polynomials
they diverge rapidly outside it — unlike the semi-empirical model, whose
physical form extrapolates smoothly.  Curie stores the fitted energy range
with the calibration and warns the first time the efficiency is evaluated
beyond it.  The Vidmar form remains the default and the recommended
choice; use the log-log form when the semi-empirical model visibly
misfits a particular detector.  The fitted model is saved with the
calibration .json as an explicit tag — necessary because a log-log
parameter array can have the same length as a Vidmar one.

Efficiency uncertainties are correlated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The efficiency measurements from a calibration source are not independent:
all points share the reference activity :math:`A_0` and decay constant
:math:`\lambda` of their source, and repeated measurements of the same
gamma line (from multiple spectra) share that line's intensity
uncertainty.  The efficiency fit therefore uses the full covariance

.. math::

   V = V_{\mathrm{stat}} + V_{\mathrm{line}} + V_{\mathrm{src}}

where :math:`V_{\mathrm{stat}}` is diagonal (counting statistics),
:math:`V_{\mathrm{line}}` couples measurements of the same line (intensity
uncertainty), and :math:`V_{\mathrm{src}}` couples all measurements of the
same source (activity and decay-constant uncertainty).  The correlated
terms are computed from the fitted curve rather than from the individual
measurements, which keeps noisy points from dragging the shared
uncertainties low.

If the reduced chi-square of the fit (computed with the full covariance
:math:`V`) exceeds one, the counting-statistics component
:math:`V_{\mathrm{stat}}` is inflated by an iterated scale factor until
:math:`\chi^2_\nu \approx 1`.  Scatter between the points is evidence
about the point-to-point uncertainties, not about the shared ones, so only
the statistical component is scaled — and, as in the peak fits, the
scaling only ever inflates.

The parameter covariance of the efficiency fit is stored with the
calibration, and `Calibration.unc_eff()` propagates it to any energy, so
efficiencies interpolated from the calibration carry correlated,
energy-dependent uncertainties.

.. [Vidmar2001] T. Vidmar, M. Korun, A. Likar and M. Lipoglavsek, "A
   physically founded model of the efficiency curve in gamma-ray
   spectrometry", *J. Phys. D: Appl. Phys.* **34** (2001) 2555,
   `doi:10.1088/0022-3727/34/16/323
   <https://doi.org/10.1088/0022-3727/34/16/323>`_.

.. [Kis1998] Z. Kis et al., "Comparison of efficiency functions for Ge
   gamma-ray detectors in a wide energy range", *Nucl. Instrum. Methods A*
   **418** (1998) 374, `doi:10.1016/S0168-9002(98)00778-5
   <https://doi.org/10.1016/S0168-9002(98)00778-5>`_.

.. _methods_reporting:

Fit Reporting Conventions
-------------------------

Every fit in Curie — the peak fits, the three calibrations, and the
decay-chain fits — reports its results, selections and problems through
two channels that share one vocabulary: console messages and per-object
diagnostics tables.

**Console messages** have the form ``[LEVEL] Class.method: message``, with
the instance identified where several may run in a loop (e.g.
``Spectrum(<filename>).fit_peaks``).  In order of escalation:

* ``ERROR`` — paired with a raised exception.
* ``WARNING`` — something that may affect results: failed fits, identical
  gammas fit with shared bounds, filters that matched nothing,
  extrapolation beyond a fitted range.
* ``INFO`` — routine accounting: fit summaries with their drop counts,
  model selections.  Nothing at INFO changes a result.
* ``DEBUG`` — the per-item detail behind every summary count (each dropped
  candidate with its reason, each parameter at a fit bound).

The configured level is a threshold: everything at that level *and above*
is shown, so the default ``'INFO'`` prints INFO, WARNING and ERROR
messages, and ``ci.set_log_level('DEBUG')`` shows everything including the
per-item detail.  ``ci.quiet_warnings()`` restricts output to errors, and
``ci.log_to('curie.log')`` also writes the session's messages to a file —
overwriting an existing file at that path (analysis scripts are typically
re-run many times) unless ``mode='append'`` (or ``'a'``) is given.

**Diagnostics tables** (``sp.diagnostics``, ``cb.diagnostics``,
``dc.diagnostics``) record one row per fit with a shared schema: the
reduced chi-square and degrees of freedom, the points used and dropped,
the model tag, the uncertainty scale factor applied (1.0 = none), a
``flags`` column from a fixed vocabulary — ``at_bound:<param>``
(a parameter ended on a fit bound), ``chi2_high`` (reduced chi-square
above 10), ``fit_failed``, ``unmoved`` (fit returned its starting
estimate), ``singular_cov`` — and the full message text.  The tables are
rebuilt on each fit and reading them never triggers one.

**Calibration point tables** (``cb.engcal_data``, ``cb.rescal_data``,
``cb.effcal_data``) present every measured calibration point with a
``used`` column and the ``reason`` a point was rejected.  Pre-fit cuts
(``unc>25%``/``unc>33%``) exclude points from the fit; post-fit outlier
clips (``outlier >3.16 sigma``) remove points from the *stored* calibration
data but not from the fit that produced it.  Nothing is silently
discarded: rejected points stay in these tables and appear as grey open
markers on the calibration plots.

**Uncertainty scale factors** follow one rule everywhere: when data are
mutually inconsistent (reduced chi-square above one), only the independent
(statistical) uncertainty component is inflated, iteratively, until the
reduced chi-square (computed with the full covariance) is consistent with
one — shared systematic components
are never scaled, and no uncertainty is ever deflated.  The applied factor
is reported in the fit's summary line and diagnostics row.

.. _methods_decay_chains:

Radioactive Decay Chains
------------------------

The Bateman equations
~~~~~~~~~~~~~~~~~~~~~

A radioactive decay chain is governed by a system of coupled first-order
differential equations: the number of atoms :math:`N_m` of each member
changes through its own decay, through the decay of its parents, and
through any external production,

.. math::

   \frac{dN_m}{dt} = R_m(t) - \lambda_m N_m
       + \sum_{j} \mathrm{BR}_{j\to m}\,\lambda_j N_j

where :math:`\lambda_m` is the decay constant, :math:`R_m(t)` the
production rate (e.g. from a nuclear reaction during irradiation), and
:math:`\mathrm{BR}_{j\to m}` the branching ratio for parent :math:`j`
decaying to :math:`m`.  `DecayChain` builds the system automatically: it
starts from the parent isotope and follows every decay branch in the
decay database — alpha, beta, electron capture, isomeric transition,
spontaneous fission — until it reaches stable nuclei.

For a linear chain :math:`1 \to 2 \to \dots \to m`, the classic solution
of Bateman [Bateman1910]_ expresses the activity
:math:`A_m = \lambda_m N_m` as a sum of exponentials.  For an initial
activity :math:`A_i(0)` of member :math:`i`,

.. math::

   A_m(t) = \sum_{i=1}^{m} A_i(0)\,\frac{\lambda_m}{\lambda_i}
       \left(\prod_{k=i}^{m-1} \mathrm{BR}_k \lambda_k\right)
       \sum_{j=i}^{m} \frac{e^{-\lambda_j t}}
       {\prod_{k \neq j} (\lambda_k - \lambda_j)}

and a constant production rate :math:`R_i` into member :math:`i`
contributes

.. math::

   A_m(t) = R_i\,\lambda_m
       \left(\prod_{k=i}^{m-1} \mathrm{BR}_k \lambda_k\right)
       \sum_{j=i}^{m} \frac{1 - e^{-\lambda_j t}}
       {\lambda_j \prod_{k \neq j} (\lambda_k - \lambda_j)}

which for a single isotope reduces to the saturation curve
:math:`A(t) = R\,(1 - e^{-\lambda t})`: during a constant irradiation the
activity builds up from zero toward the saturation value :math:`R`, where
production and decay balance.
When the decay graph branches, Curie enumerates
every distinct path from the parent to the isotope of interest and sums
the contributions, each weighted by its product of branching ratios.

Production schedules are piecewise constant: within each time interval
the production rates are held fixed and the solution above is applied,
with the activities at the end of one interval becoming the initial
activities of the next.  When a production schedule is given, time zero
is the *end* of production (the end of bombardment, for activation
experiments); for a chain specified only by initial activities, time zero
is the moment those activities held.  All subsequent times — counting
intervals, activities — are measured from that point.

Numerical treatment
~~~~~~~~~~~~~~~~~~~

The Bateman denominators :math:`\prod_{k \neq j}(\lambda_k - \lambda_j)`
diverge when two members of a chain have (nearly) equal decay constants —
which genuinely happens in the decay database, where two half-lives can
be identical as rounded.  Rather than perturbing the values, Curie groups
decay constants that agree to within one part in :math:`10^9` and
evaluates the exact *confluent* form of the solution for each group (the
mathematical limit of the Bateman coefficients as the constants
coincide), computed with a recursion that avoids the catastrophic
cancellation of the raw sum.  Each branch is additionally rescaled by the
geometric mean of its decay constants — the solution is invariant under
this rescaling — so that long chains with widely separated half-lives
remain within floating-point range in any choice of time units.

Fitting to measured decays
~~~~~~~~~~~~~~~~~~~~~~~~~~

`fit_R()` and `fit_A0()` adjust one scalar multiplier per produced
isotope — scaling the production-rate history :math:`R(t)` (its shape is
preserved) or the initial activity — to match the observed number of
decays in each counting interval.  The fit compares *decays per
interval*, not instantaneous activities, so counts integrated over long
measurements are treated exactly.  Where the count data come from fitted
spectra (`get_counts()`), each measurement's uncertainty is decomposed
into an independent counting part, a part shared by all measurements of
the same gamma line (its intensity), and a part shared by all
measurements using the same efficiency calibration; the fit is a
generalized least squares with the resulting covariance, using the same
one-sided scale-factor convention as the calibration fits (see
:ref:`methods_calibration`).

.. [Bateman1910] H. Bateman, "The solution of a system of differential
   equations occurring in the theory of radio-active transformations",
   *Proc. Cambridge Philos. Soc.* **15** (1910) 423.

.. _methods_reaction_data:

Nuclear Reaction Data
---------------------

The libraries
~~~~~~~~~~~~~

Curie distributes cross sections from four evaluated libraries:

=================  ==========================  =========  ==============  =============
Name               Evaluation                  Particles  Organization    Uncertainties
=================  ==========================  =========  ==============  =============
``endf``           ENDF/B-VIII.1 [ENDF]_       n          exclusive       no
``tendl``          TENDL-2025 [TENDL]_         n          exclusive       no
``tendl_n/p/d/a``  TENDL-2025 [TENDL]_         n, p, d,   residual        no
                                               a
``irdff``          IRDFF-II [IRDFF]_           n          exclusive       yes
``iaea``           IAEA Medical Monitors       n, p, d,   residual        yes
                   (2025) [IAEA]_              a, h, g
=================  ==========================  =========  ==============  =============

An *exclusive* library indexes reactions by their outgoing channel —
(n,2n), (n,p), (n,inl) (inelastic scattering) — while a
*residual-product* library indexes by
what nucleus is produced, written (p,x), summing all routes.  Independent
of that split, a residual-product cross section may be *independent*
(direct production of the state only — TENDL's convention, each isomer a
separate entry) or *cumulative* (including decay feeding from short-lived
co-produced parents — common in the IAEA monitor evaluations, which
distinguish the two where both are useful).  IRDFF-II is a dosimetry
standard: most of its entries are exclusive channels, with a few indexed
by product.

The TENDL residual-product libraries also carry *natural-element*
targets (``natFe``, ``natTi``, ...): abundance-weighted sums of the
isotopic tables, computed when the databases are built from the same
decay-data generation's abundances.  Elements whose lightest isotopes
TENDL does not evaluate (H, He, Li, Be) have no natural entry.

Energies are in MeV and cross sections in mb throughout (converted on
retrieval from any source library that uses eV and barns).  On
retrieval, energy grids are sorted and exact duplicate points dropped;
duplicated energy values that encode step discontinuities (e.g.
threshold steps) are preserved for all libraries except TENDL, whose
point-wise smooth model output makes a repeated energy value a build
artifact, so there they are removed.

Interpolation and flux averages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Reaction.interpolate()` selects its scheme per library, through the
`Reaction.interp_config` parameter ``'interpolation'``.  The
pointwise-linearized libraries (ENDF — reconstructed to a 0.5 %
linearization tolerance — IRDFF and IAEA) default to ``'linear'``, the
convention their tabulated grids are generated for.  The TENDL
libraries, whose smooth model calculations sit on a coarse energy grid,
default to ``'pchip-sqrt'``: monotone piecewise-cubic (PCHIP)
interpolation carried out in :math:`\sqrt{E}`–:math:`\sqrt{\sigma}`
space.  The monotone construction passes exactly through the evaluated
points and cannot overshoot or ring between them (a conventional
quadratic or cubic spline can, badly, across a sharp threshold rise),
and the square-root energy axis linearizes the
:math:`\sigma \propto \sqrt{E - E_{thr}}` turn-on just above
threshold.  Interpolants are clamped non-negative, and below the
reaction threshold the cross section is zero.  **Outside the evaluated
energy range the interpolated cross section is zero** — Curie never
extrapolates evaluated data.  Either scheme can be selected per
reaction: ``rx.interpolate(E, interpolation='linear')``.

Flux integrals treat the energy points as bin centers, with bin edges at
the midpoints between them; the first and last points take the full
spacing to their single neighbor:

.. math::

   \langle\sigma\rangle_\phi =
     \frac{\sum_i \sigma(E_i)\,\phi_i\,\Delta E_i}
          {\sum_i \phi_i\,\Delta E_i}
   \qquad
   \Delta E_i = \tfrac{1}{2}(E_{i+1} - E_{i-1}),\quad
   \Delta E_0 = E_1 - E_0,\quad
   \Delta E_{n-1} = E_{n-1} - E_{n-2}

For a uniform grid the widths cancel from the average, and
histogram-like fluxes (such as those from `Stack`) are integrated
without the endpoint under-weighting a trapezoidal rule would introduce.
The cross-section uncertainty of an average is propagated as fully
correlated between energies — uncertainties are summed linearly, not in
quadrature — which is the conservative choice for evaluated curves,
whose errors are dominated by common normalization rather than
point-to-point scatter.

.. _methods_stopping:

Stopping Power and Particle Transport
-------------------------------------

Stopping powers
~~~~~~~~~~~~~~~

A charged particle moving through matter loses energy continuously, at a
rate characterized by the stopping power :math:`S = -dE/dx`.  Curie uses
the semi-empirical parameterization of Andersen and Ziegler
[AZ1977]_ [Z1977]_: the electronic stopping power for protons in each
element is a fitted piecewise form — a velocity-proportional regime at
the lowest energies, an intermediate form joining smoothly to it, and a
Bethe-like form with fitted corrections at high energy — with tabulated
coefficients for every element up to uranium.  Alpha particles have
their own coefficient set; deuterons and tritons use the proton stopping
evaluated at the same velocity (i.e. scaled energy :math:`E/M`).  For
heavier ions at low-to-moderate velocity, the proton stopping is scaled
by the square of an effective charge ratio (a Bohr/Northcliffe-type
parameterization) that accounts for partial neutralization of the ion;
above roughly 1 MeV per nucleon a relativistic Bethe-like form is used
directly.  A nuclear-stopping term, significant only at the lowest
energies, is added in all cases.

The stopping power of a compound is the mass-weighted sum of its
elemental stopping powers (Bragg additivity), which neglects the
influence of chemical bonds on the electronic structure — typically a
percent-level approximation, largest for light compounds at low
energies.  The range of a particle is the continuous-slowing-down
integral

.. math::

   R(E_0) = \int_0^{E_0} \frac{dE}{S(E)}

Photon attenuation coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Element` and `Compound` also carry photon interaction data: the
mass-attenuation coefficient :math:`\mu/\rho` and the mass
energy-absorption coefficient :math:`\mu_{en}/\rho` are log-log
interpolations of the NIST XCOM tabulations, with compound values
likewise combined by mass-weighted additivity.

Transport through a stack
~~~~~~~~~~~~~~~~~~~~~~~~~

`Stack` propagates an ensemble of particles whose initial energies are
Monte Carlo-sampled from a Gaussian of mean ``E0`` and width ``dE0``
(default 1% of ``E0``); the transport itself is then deterministic,
foil by foil in beam order.  Within each foil the
energy loss is integrated by a predictor–corrector (Heun) scheme: a
trial step with the stopping power at the current energy, corrected by
the average of the stopping powers at the start and trial energies.  The
number of steps is set so that the expected fractional energy loss per
step is approximately the ``accuracy`` argument (default 0.01, i.e.
about 1% of the current energy lost per step), clipped to the range
``min_steps`` to ``max_steps`` (default 2 to 50): a foil that barely
degrades the beam is integrated in the minimum number of steps, a thick
degrader in more.  The energy
distribution — the "flux" — assigned to a foil is the histogram of the
ensemble's energies over all integration steps inside that foil, i.e. a
path-averaged distribution through the foil's thickness, from which the
reported mean energy ``mu_E`` and width ``sig_E`` are the first two
moments.  Because each integration step advances the beam by an equal
increment of areal density, tallying the ensemble's energy at every step
weights each energy by the path length the beam spends there.  That
path-length weighting is, up to normalization, the particle fluence —
track length per unit volume — as a function of energy, which is
precisely the spectral weight a reaction rate integrates over:
convolving this distribution with an excitation function (the flux
average of :ref:`methods_reaction_data`) yields the effective cross
section the foil actually experiences.  ``mu_E`` is thus the
fluence-weighted mean energy of the beam within the foil.

The width ``sig_E`` of a foil's distribution has two distinct physical
origins.  The first is the energy the beam loses between the front and
back faces of the foil: because the histogram is accumulated over the
entire path through the foil, a thick degrader that lowers the beam by
several MeV produces a broad distribution however narrow the incident
beam was, and for such a foil this within-foil loss — not the beam
spread — dominates ``sig_E``.  The second is the incident beam spread
``dE0`` together with its growth as the ensemble slows: slower particles
lose energy faster, so an initially narrow beam diverges in energy as it
degrades.  ``sig_E`` is therefore the physical range of energies the
foil samples, and it is the energy uncertainty to attach to a cross
section measured in that foil — not the statistical uncertainty of the
mean ``mu_E``.  Collisional energy-loss
straggling — the stochastic variance growth described by Bohr and
Tschalar — is *not* added, so for very thick degraders the true energy
spread is somewhat larger than computed.  Particles are neither absorbed
nor deflected: the distributions describe the beam's energy, not its
intensity, and lateral spread is not modeled.

Because the beam is degraded foil by foil, uncertainties in ``E0``, the
areal densities, and the stopping powers accumulate down the stack, so
the energy assigned to a deep foil can be off by more than its computed
width.  In stacked-target work this is corrected empirically by
including monitor foils with independently evaluated cross sections and
adjusting ``E0`` and the overall density multiplier ``dp`` until the
beam current inferred from each monitor reaction varies smoothly along
the stack.  `Stack` does not perform this fit itself; it exposes ``E0``
and ``dp`` so the transport can be repeated inside a user-driven
minimization.

.. [AZ1977] H.H. Andersen and J.F. Ziegler, *Hydrogen: Stopping Powers
   and Ranges in All Elements* (Pergamon, New York, 1977).

.. [Z1977] J.F. Ziegler, *Helium: Stopping Powers and Ranges in All
   Elemental Matter* (Pergamon, New York, 1977).

.. _methods_provenance:

Data Provenance and Integrity
-----------------------------

Curie's nuclear data ship as pre-built databases, fetched on first use
and verified against published SHA256 checksums (see
:ref:`quickinstall`); the large cross-section libraries are fetched in
per-target pieces.  The reaction libraries are described in
:ref:`methods_reaction_data`.  The decay data (half-lives, branching
ratios, emission energies and intensities) are compiled from ENSDF via
the IAEA LiveChart interface, with NUBASE2020/AME2020 masses and
half-life closures and ENDF/B-VIII.1 fission yields (see
:ref:`data_sources` for full provenance); photon attenuation
coefficients derive from the NIST XCOM tabulations; charged-particle
stopping powers use the Andersen–Ziegler formulation.  Gamma-ray
energies are in keV in the spectroscopy classes, and every half-life is
in seconds unless a unit argument says otherwise.

.. [ENDF] "ENDF/B-VIII.1", National Nuclear Data Center, Brookhaven
   National Laboratory (2024), https://www.nndc.bnl.gov/endf-b8.1/.

.. [TENDL] A.J. Koning and D. Rochman, "Modern Nuclear Data Evaluation
   with the TALYS Code System", *Nucl. Data Sheets* **113** (2012) 2841;
   TENDL-2025 files from https://tendl.imperial.ac.uk (CC BY 4.0).

.. [IRDFF] A. Trkov et al., "IRDFF-II: A New Neutron Metrology Library",
   *Nucl. Data Sheets* **163** (2020) 1.

.. [IAEA] A. Hermanne et al., "Reference Cross Sections for
   Charged-particle Monitor Reactions", *Nucl. Data Sheets* **148**
   (2018) 338.
