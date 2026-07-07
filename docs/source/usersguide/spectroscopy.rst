.. _spectroscopy:

============
Spectroscopy
============

Curie provides two classes for analyzing gamma-ray spectra from high-purity
germanium (HPGe) detectors: the `Spectrum` class, which reads spectra and
performs peak fitting, and the `Calibration` class, which generates and
stores the energy, resolution and efficiency calibrations needed to convert
fitted peaks into decay rates and activities.

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
3. **Calibrate**: apply a saved calibration (``sp.cb = 'calib.json'``), or
   fit one from spectra of reference sources with
   `Calibration.calibrate()`.
4. **Fit** the peaks: ``sp.fit_peaks()``, tuned by the ``fit_config``
   options.  The result is the ``sp.peaks`` table of counts, decays and
   decay rates per gamma line.
5. **Inspect and export**: ``sp.summarize()``, ``sp.plot()``, and
   ``sp.saveas()`` to .csv, .json, .db or .Chn.

Each of these steps is detailed in the :ref:`spectroscopy_tasks` page, the
:ref:`spectroscopy_tutorial` walks through a complete efficiency
calibration with a :sup:`152`\ Eu source, and
:ref:`spectroscopy_troubleshooting` collects the most common pitfalls —
most of them involving the energy calibration.

Uses and limitations
--------------------

These classes are designed for *activation analysis*: quantifying the decay
rates of known gamma-emitting isotopes in a counted sample.  The peak-fit
model (Gaussian with optional skew and step components on a SNIP or
polynomial background, see :ref:`methods_peak_fitting`) and the
semi-empirical efficiency model (see :ref:`methods_calibration`) are
well-suited to HPGe data; they are not intended for low-resolution
detectors such as NaI.  Peak identification is driven by the assigned
isotope list — Curie fits the lines it expects to find, rather than
searching for unknown peaks.  True-coincidence summing corrections are not
currently applied, so efficiency calibrations should be performed at
geometries where summing is negligible, or with single-line sources.

.. toctree::
   :maxdepth: 1

   spectroscopy_tasks
   spectroscopy_tutorial
   spectroscopy_troubleshooting
