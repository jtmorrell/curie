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

Curie models a gamma-ray peak in an HPGe spectrum as a Gaussian with an
optional low-energy skew component and an optional step in the background,
as a function of ADC channel number :math:`x`:

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
Compton continuum on either side of the peak.  By default :math:`R=0.1`,
:math:`\alpha=0.9` and :math:`\mathrm{step}=0`, and these three parameters
are held fixed; the ``skew_fit`` and ``step_fit`` options of
`Spectrum.fit_peaks()` add them to the fitted parameters instead.

Background
~~~~~~~~~~

The continuum under the peaks is modeled in one of two ways, selected by the
``bg`` option:

* A polynomial — constant, linear or quadratic in channel number — fit
  jointly with the peaks in each multiplet.

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
parameter covariance is taken directly from the fit, without the customary
rescaling by :math:`\chi^2_\nu` that would shrink the uncertainties of
clean peaks below the floor set by counting statistics.  When the reduced
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
linear (or optionally quadratic) polynomial,

.. math::

   E(x) = a_0 + a_1 x \;(+\, a_2 x^2)

fit by weighted least squares to the fitted peak centroids versus the known
line energies, weighted by the centroid uncertainties.  Points with energy
uncertainties larger than 25% of the energy are excluded.

Resolution calibration
~~~~~~~~~~~~~~~~~~~~~~

The peak width (the Gaussian :math:`\sigma`, in channels) is modeled as
either a linear function of channel number (the default) or a square-root
function,

.. math::

   \sigma(x) = b_0 + b_1 x
   \qquad\text{or}\qquad
   \sigma(x) = b_0 \sqrt{x}

fit to the fitted peak widths.  The square-root form is the expectation
from pure counting statistics of charge-carrier generation; the linear form
accounts for the additional electronic-noise contribution typical of real
HPGe systems.  Strongly discrepant points (squared residual more than ten
times the variance) are excluded from the retained calibration data.

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
kept.  The interaction coefficients are log-log interpolations of XCOM
tabulations.

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
:math:`A(t) = R\,(1 - e^{-\lambda t})` familiar from activation work.
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
blow up when two members of a chain have (nearly) equal decay constants —
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

================  ==========================  =========  ==============  =============
Name              Evaluation                  Particles  Organization    Uncertainties
================  ==========================  =========  ==============  =============
``endf``          ENDF/B-VII.1 [ENDF]_        n          exclusive       no
``tendl``         TENDL-2015 [TENDL]_         n          exclusive       no
``tendl_n/p/d``   TENDL-2015 [TENDL]_         n, p, d    residual        no
``irdff``         IRDFF-II [IRDFF]_           n          exclusive       yes
``iaea``          IAEA CP-Reference           n, p, d,   residual        yes
                  (2017) [IAEA]_              a, h, g
================  ==========================  =========  ==============  =============

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

Energies are in MeV and cross sections in mb throughout (converted on
retrieval from any source library that uses eV and barns).  On
retrieval, energy grids are sorted and exact duplicate points dropped;
duplicated energy values that encode step discontinuities (e.g.
threshold steps) are preserved for all libraries except TENDL, whose
point-wise smooth model output makes a repeated energy value a build
artifact, so there they are removed.

Interpolation and flux averages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Reaction.interpolate()` is piecewise-linear on the evaluated grid, with
two exceptions: TENDL curves, which are smooth model calculations, are
interpolated quadratically (from just below the reaction threshold, and
falling back to linear for very short grids), and negative interpolants
are clamped to zero.  **Outside the evaluated energy range the
interpolated cross section is zero** — Curie never extrapolates
evaluated data.

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
the average of the stopping powers at the start and trial energies, with
the number of steps chosen adaptively from the expected fractional
energy loss (bounded by ``min_steps`` and ``max_steps``).  The energy
distribution — the "flux" — assigned to a foil is the histogram of the
ensemble's energies over all integration steps inside that foil, i.e. a
path-averaged distribution through the foil's thickness, from which the
reported mean energy ``mu_E`` and width ``sig_E`` are the first two
moments.

The width of these distributions reflects the initial beam spread and
the increasing divergence of particle energies as the ensemble slows
(slower particles lose energy faster).  Collisional energy-loss
straggling — the stochastic variance growth described by Bohr and
Tschalar — is *not* added, so for very thick degraders the true energy
spread is somewhat larger than computed.  Particles are neither absorbed
nor deflected: the distributions describe the beam's energy, not its
intensity, and lateral spread is not modeled.

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
ratios, emission energies and intensities) are compiled from NuDat 2.0,
ENDF/B-VII.0 and the nuclear wallet cards; photon attenuation
coefficients derive from the NIST XCOM tabulations; charged-particle
stopping powers use the Andersen–Ziegler formulation.  Gamma-ray
energies are in keV in the spectroscopy classes, and every half-life is
in seconds unless a unit argument says otherwise.

.. [ENDF] M.B. Chadwick et al., "ENDF/B-VII.1 Nuclear Data for Science
   and Technology", *Nucl. Data Sheets* **112** (2011) 2887.

.. [TENDL] A.J. Koning and D. Rochman, "Modern Nuclear Data Evaluation
   with the TALYS Code System", *Nucl. Data Sheets* **113** (2012) 2841.

.. [IRDFF] A. Trkov et al., "IRDFF-II: A New Neutron Metrology Library",
   *Nucl. Data Sheets* **163** (2020) 1.

.. [IAEA] A. Hermanne et al., "Reference Cross Sections for
   Charged-particle Monitor Reactions", *Nucl. Data Sheets* **148**
   (2018) 338.
