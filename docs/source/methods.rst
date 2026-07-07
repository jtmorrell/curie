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
and width of the Gaussian.  The second term is a skewed-Gaussian tail on the
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
parameter covariance is taken directly from the fit without rescaling by
:math:`\chi^2_\nu` — rescaling would deflate the uncertainties of clean
peaks below the counting-statistics floor.  Instead, when the reduced
chi-square (evaluated over the non-empty channels) exceeds one, the
covariance is inflated by :math:`\chi^2_\nu` — a one-sided scale factor
convention: model mismatch can only increase the uncertainty, never reduce
it below counting statistics.

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
same source (activity and decay-constant uncertainty).  Correlated
magnitudes are computed from the fitted model values rather than the
measured ones, which avoids the downward bias of measurement-weighted
correlated fits.  If the whitened reduced chi-square exceeds one, the
*independent* (statistical) component is inflated by an iterated scale
factor until :math:`\chi^2_\nu \approx 1` — inconsistency between points
cannot indict the correlated modes, so only the independent component is
scaled, and (as everywhere in Curie) the scale factor is one-sided: it
inflates, never deflates.

The parameter covariance of the efficiency fit is stored with the
calibration, and `Calibration.unc_eff()` propagates it to any energy, so
efficiencies interpolated from the calibration carry correlated,
energy-dependent uncertainties.

.. [Vidmar2001] T. Vidmar, M. Korun, A. Likar and M. Lipoglavšek, "A
   physically founded model of the efficiency curve in gamma-ray
   spectrometry", *J. Phys. D: Appl. Phys.* **34** (2001) 2555,
   `doi:10.1088/0022-3727/34/16/323
   <https://doi.org/10.1088/0022-3727/34/16/323>`_.
