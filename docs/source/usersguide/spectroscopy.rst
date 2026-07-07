.. _spectroscopy:

============
Spectroscopy
============

Curie provides two classes for analyzing gamma-ray spectra from high-purity
germanium (HPGe) detectors: the `Spectrum` class, which reads spectra and
performs peak fitting, and the `Calibration` class, which generates and
stores the energy, resolution and efficiency calibrations needed to convert
fitted peaks into activities.

.. figure:: ../images/eu_spectrum_fit.png
   :width: 100%

   A :sup:`152`\ Eu calibration spectrum with fitted peaks.

Workflow
--------

A typical spectroscopy analysis proceeds in five steps:

1. **Load** a spectrum from disk: ``sp = ci.Spectrum('eu_calib_7cm.Spe')``.
   Ortec .Spe and .Chn and Canberra .CNF and .IEC formats are supported.
2. **Identify** the gamma-decaying isotopes present:
   ``sp.isotopes = ['152EU', '40K']``.  Curie retrieves their gamma lines
   from its decay data.
3. **Calibrate**: apply a saved calibration for this detector and counting
   geometry: ``sp.cb = 'eu_calib.json'``.
4. **Fit** the peaks: ``sp.fit_peaks()``, tuned by the ``fit_config``
   options.  The result is the ``sp.peaks`` table of counts, decays and
   activities per gamma line.
5. **Inspect and export**: ``sp.summarize()``, ``sp.plot()``, and
   ``sp.saveas()`` to .csv, .json, .db or .Chn.

These steps describe routine analysis with an existing calibration.
*Creating* that calibration is a separate, usually one-time task:
`Calibration.calibrate()` fits the peaks of reference-source spectra
itself, so it runs before any manual peak fitting — the
:ref:`spectroscopy_tutorial` walks through it.

Each of these steps is detailed in the :ref:`spectroscopy_tasks` page, the
:ref:`spectroscopy_tutorial` walks through a complete efficiency
calibration with a :sup:`152`\ Eu source, and
:ref:`spectroscopy_troubleshooting` collects the most common pitfalls —
most of them involving the energy calibration.

Uses and limitations
--------------------

These classes are designed for *activation analysis*: quantifying the
activities of known gamma-emitting isotopes in a counted sample.  The
peak-fit and efficiency models (described in :ref:`methods_peak_fitting`
and :ref:`methods_calibration`) are tuned for HPGe data, and are not
intended for low-resolution detectors such as NaI.  Peak identification is
driven by the assigned isotope list — Curie fits the gamma lines it
expects from those isotopes, rather than hunting for unidentified peaks.
True-coincidence summing — two gammas from the same decay cascade arriving
together and counted as one event — is not corrected; it matters mainly
for samples counted very close to the detector, so calibrate at a moderate
standoff or with single-line sources.

.. toctree::
   :maxdepth: 1

   spectroscopy_tasks
   spectroscopy_tutorial
   spectroscopy_troubleshooting
