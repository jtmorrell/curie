.. _spectroscopy_troubleshooting:

============================
Spectroscopy Troubleshooting
============================

The most common spectroscopy problems trace back to one thing: Curie fits
the peaks it *expects* from the assigned isotopes and the current
calibration, rather than searching the spectrum for unknown peaks.  When
the expectation is wrong — usually the energy calibration — peaks are
missed or misfit.  This page collects the failure modes we see most often,
with their symptoms and fixes.

Whatever the symptom, check the fit's output first: the ``fit_peaks``
summary line accounts for every candidate line (fit, or dropped with the
responsible filter named), and ``sp.diagnostics`` records each attempted
fit with flags for the problematic ones (see :ref:`spectroscopy_howto`).
Most of the entries below appear as a line item in one or the other.

No peaks are fit, and the spectrum looks uncalibrated
-----------------------------------------------------

**Symptom:** ``sp.peaks`` comes back empty (or nearly so) even though the
spectrum plainly shows peaks, and known lines are nowhere near their
energies in ``sp.plot()``.

**Cause:** the energy calibration stored in the spectrum file is wrong or
missing.  Curie predicts the channel of every gamma line from the energy
calibration and evaluates its expected signal-to-noise ratio *at that
channel*; if the calibration points at featureless background, every line
fails the ``SNR_min`` test and is dropped without an error.

**Fix:** plot the raw histogram against channel number and identify one or
two known lines by eye::

	sp.plot(fit=False, xcalib=False)

then anchor the calibration with (channel, energy) pairs — here using the
121.8 keV line of :sup:`152`\ Eu seen at channel 664::

	sp.isotopes = ['152EU']
	sp.auto_calibrate(peaks=[[664, 121.8]])

or set it directly if the coefficients are known::

	sp.cb.engcal = [0.3, 0.184]
	sp.fit_peaks()

Peaks fit correctly — until a saved calibration is applied
----------------------------------------------------------

**Symptom:** the spectrum fits well on its own, but after assigning a saved
calibration (``sp.cb = 'eu_calib.json'``) the peaks shift or disappear.
This one is trickier, because applying the calibration is exactly what you
are supposed to do.

**Cause:** assigning a calibration replaces *all three* calibrations —
energy, resolution and efficiency.  The efficiency calibration is usually
what you want from the file (it is a property of the detector and counting
geometry), but the energy calibration it carries describes the electronics
*on the day the calibration was measured*.  Drift in the amplifier gain
since then (or an intervening, forgotten recalibration of the ADC) means
the file's energy
calibration no longer matches this spectrum, while the spectrum's own
header calibration was correct all along.

**Fix:** apply the saved calibration for its efficiency, but keep the
spectrum's own energy calibration::

	sp = ci.Spectrum('sample_7cm.Spe')
	engcal = sp.cb.engcal          # calibration from the file header
	sp.cb = 'eu_calib.json'        # detector efficiency (overwrites engcal)
	sp.cb.engcal = engcal          # restore this spectrum's energy calibration
	sp.fit_peaks()

If the header calibration is also imperfect, follow with
``sp.auto_calibrate()``, which makes small corrections using the assigned
isotopes' lines.

An expected peak is missing from the results
--------------------------------------------

**Symptom:** a line you can see in the spectrum — and that you know the
isotope emits — does not appear in ``sp.peaks``.  Everything else fits
fine.

**Cause:** one of the peak-selection filters excluded it before fitting.
The exclusion is reported: the summary line counts the dropped lines per
filter, and ``ci.set_log_level('DEBUG')`` names each one individually::

	[INFO] Spectrum(sample.Spe).fit_peaks: fit 12 peaks in 9 multiplets from
	1 isotopes; dropped 31 candidates (2 SNR<4.0, 29 intensity<0.05%)

The defaults are chosen for typical activation spectra and will drop some
legitimate lines:

* ``E_min`` (default 75 keV) excludes all lower-energy lines — the 59.5 keV
  line of :sup:`241`\ Am, for example, never fits at the default.
* ``I_min`` (default 0.05%) excludes weak branches.
* ``dE_511`` excludes lines within a few keV of the 511 keV annihilation
  peak.
* ``SNR_min`` excludes lines *predicted* to be too weak to fit — and the
  prediction uses the current efficiency calibration, so a badly wrong
  efficiency curve can veto a clearly visible peak.
* X-rays are excluded entirely unless ``xrays=True``.

**Fix:** relax the relevant filter::

	sp.fit_peaks(E_min=40.0, I_min=0.01)   # admit lower-energy / weaker gammas
	sp.fit_peaks(xrays=True, E_min=20.0)   # or: also include x-ray lines
	sp.fit_peaks(SNR_min=2.0)              # or: admit marginal peaks

Each line is an independent alternative, not a sequence.  If a *clearly
visible* peak is being vetoed by ``SNR_min``, the real problem is usually
an efficiency calibration that badly underpredicts the peak — recalibrate
(see the :ref:`spectroscopy_examples`) rather than lowering the threshold.
If a line is missing because the isotope's decay data doesn't include it,
add it manually with the ``gammas`` argument (see
:ref:`spectroscopy_howto`).

Fits succeed, but they are bad fits
-----------------------------------

**Symptom:** fitted curves that visibly miss the data, large ``chi2``
values in the peak table, or ``peak fit failed`` warnings for particular
multiplets (groups of overlapping peaks that are fit together).  Failed
multiplets are crosshatched in red on ``sp.plot()``, and every high-chi2
or at-bound fit is flagged in ``sp.diagnostics``.

**Cause and fix, by situation:**

* **The peak sits on structure** — the peak sits on a broad spectral
  feature rather than flat background: a backscatter peak or Compton edge
  (both produced by scattered gamma rays), or the shoulder of a much
  larger neighboring peak.  The default SNIP background assumes
  a smoothly varying continuum and will not follow sharp features.  Refit
  with a polynomial background, which is fit jointly with the peaks:
  ``sp.fit_peaks(bg='quadratic')``.

* **The fit window is too wide or too crowded.**  Wide windows
  (``pk_width``, default 7.5 peak widths) can pull neighboring structure
  into the fit, and crowded regions can exceed the multiplet limit.
  Reduce ``pk_width``, or raise ``multi_max`` so overlapping peaks are fit
  together rather than truncated.

* **The peak shape doesn't match** — strong low-energy tailing on intense
  peaks (degraded detectors, high count rates) that the default fixed skew
  parameters underestimate.  Let the skew be fit per peak with
  ``skew_fit=True`` (and ``step_fit=True`` for a per-peak step), at the
  cost of more free parameters per peak.

* **The starting point is too constrained.**  Amplitudes, centroids and
  widths are bounded around their predicted values; if the prediction is
  poor (e.g. a marginal energy calibration), loosen the bounds with the
  multipliers ``A_bound``, ``mu_bound`` and ``sig_bound``.

A ``chi2`` well above one on a very high-statistics peak — values of ten
or more are common on peaks with hundreds of thousands of counts, like the
122 keV peak in the :ref:`spectroscopy_examples` — is not by itself a bad
fit: with that many counts, even percent-level imperfections of the peak
model are statistically resolvable.  Curie accounts for this by inflating
the fit uncertainties by :math:`\sqrt{\chi^2_\nu}` (see
:ref:`methods_peak_fitting`).
