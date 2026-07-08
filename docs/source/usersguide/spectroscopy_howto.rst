.. _spectroscopy_howto:

=========================
Spectroscopy How-to Guide
=========================

This page is a task-by-task reference for the `Spectrum` and `Calibration`
classes.  All examples assume ``import curie as ci``.  For the underlying
models and formulas, see :ref:`methods_peak_fitting` and
:ref:`methods_calibration`; for a complete worked example, see the
:ref:`spectroscopy_examples`.

Loading spectra
---------------

`Spectrum` reads Ortec .Spe (ASCII) and .Chn (binary), and Canberra .CNF
and .IEC files::

	sp = ci.Spectrum('eu_calib_7cm.Spe')

The parsed data are available as attributes: ``sp.hist`` (the channel
histogram), ``sp.start_time``, ``sp.live_time`` and ``sp.real_time``.
Spectra of the same length can be summed with ``+``, and rebinned with
``sp.rebin(N_bins)``.

Identifying peaks
-----------------

Assign the isotopes present in the sample, using Curie's isotope naming
convention (e.g. ``'152EU'``, ``'Eu-152'``, ``'115INm'``)::

	sp.isotopes = ['152EU']

Curie generates the list of gamma lines to fit from the decay data of these
isotopes.  Lines can also be given directly to `fit_peaks` — for example a
line from an isotope you have not assigned, or one missing from the decay
data — with energies in keV and intensities in percent::

	sp.fit_peaks(gammas=[{'energy':1460.82, 'intensity':10.66,
	                      'unc_intensity':0.17, 'isotope':'40K'}])

Fitting peaks
-------------

``sp.fit_peaks()`` fits all selected lines and returns the peak table; it
is also called automatically the first time ``sp.peaks``, ``sp.plot()``,
``sp.summarize()`` or ``sp.saveas()`` is used.  The fits are cached: if you
change the calibration, isotope list or fit options afterwards, call
``sp.fit_peaks()`` again to re-fit.

``sp.peaks`` is a pandas DataFrame with one row per fitted gamma line.  Its
central columns are:

================== ============================================================
Column             Meaning
================== ============================================================
``isotope``        The emitting isotope
``energy``         Gamma-line energy, in keV
``counts``         Net counts in the peak (background subtracted)
``intensity``      Gamma intensity :math:`I_\gamma` (branching ratio)
``efficiency``     Peak efficiency at the line energy, from the calibration
``decays``         Decays of the isotope during the count
``decay_rate``     Average decay rate during the count — i.e. the activity,
                   in Bq (``summarize()`` prints this value as "activity")
``chi2``           Reduced chi-square of the multiplet fit
================== ============================================================

Each quantity is paired with an ``unc_`` column giving its absolute
uncertainty.  ``decays`` and ``decay_rate`` are corrected for dead time,
efficiency, intensity, and any attenuation/geometry corrections that have
been applied.

Configuring the fit
-------------------

Fit options are set through the ``fit_config`` dictionary — as an
attribute (``sp.fit_config = {...}``, persistent) or as keyword arguments
to ``sp.fit_peaks(...)``.  The complete reference for every option is the
`Spectrum.fit_peaks()` API entry; they fall into three groups:

**Peak-shape and background options** — ``bg`` selects the background
model (``'snip'`` default, or ``'constant'``/``'linear'``/``'quadratic'``)
and ``snip_adj`` scales the SNIP background parameters; ``R``, ``alpha``
and ``step`` set the skew and step components of the peak shape, and
``skew_fit``/``step_fit`` control whether they are fit per peak or held
fixed (see :ref:`methods_peak_fitting` for the functional forms).

**Peak-selection options** — ``xrays``, ``E_min``, ``I_min`` and
``dE_511`` filter the candidate gamma lines; ``SNR_min`` drops lines whose
predicted signal-to-noise ratio is too low to fit reliably; ``ident_idx``
controls how overlapping (identical-energy) lines are merged or flagged.

**Fit-window and bound options** — ``pk_width`` sets the fitted window
around each peak; ``multi_max`` limits the number of peaks fit together as
a multiplet; ``A_bound``, ``mu_bound`` and ``sig_bound`` scale the bounds
on the amplitude, centroid and width parameters.

For example, to include x-ray lines down to 20 keV on a quadratic
background::

	sp.fit_config = {'xrays':True, 'E_min':20.0, 'bg':'quadratic'}

Calibrating
-----------

A `Calibration` holds three calibrations (see :ref:`methods_calibration`
for the functional forms and fitting procedure):

* **energy** — ``cb.engcal``, channel to keV, used by ``cb.eng()``;
* **resolution** — ``cb.rescal``, peak width vs. channel, used by
  ``cb.res()``;
* **efficiency** — ``cb.effcal`` with covariance ``cb.unc_effcal``,
  used by ``cb.eff()`` and ``cb.unc_eff()``.

To fit all three from spectra of reference sources, give the source
activities at a reference date::

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']

	cb = ci.Calibration()
	cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.7E4,
	                             'ref_date':'01/01/2009 12:00:00'}])
	cb.plot()

``sources`` can also be a .csv, .json or .db file with columns 'isotope',
'A0' and 'ref_date' (an 'unc_A0' column is used if present).  The
calibration is applied to the input spectra, and can be saved and re-used
across spectra counted on the same detector and geometry::

	cb.saveas('eu_calib.json')

	sp2 = ci.Spectrum('sample_7cm.Spe')
	sp2.cb = 'eu_calib.json'

The energy calibration can also be set directly (``sp.cb.engcal = [0.3,
0.184]``) — after changing it, re-fit with ``sp.fit_peaks()``.

Adjusting a drifted energy calibration
--------------------------------------

If the stored energy calibration is slightly off (peaks visibly displaced
from their lines), ``sp.auto_calibrate()`` re-fits it using a forward-fit
of the assigned isotopes' lines to the spectrum.  It only converges from a
starting point within about half a percent; for larger drifts, give it a
guess or a list of (channel, energy) anchor points::

	sp.auto_calibrate(guess=[0.3, 0.1835])
	sp.auto_calibrate(peaks=[[664, 121.8]])

Correcting for attenuation and geometry
---------------------------------------

Two multiplicative corrections can be applied to a spectrum before fitting;
both modify the ``decays``/``decay_rate`` columns of the peak table.

``sp.attenuation_correction(compounds, x=...)`` computes the
energy-dependent self-attenuation of the sample: the first entry is the
sample itself (its correction accounts for emission throughout the
thickness), and subsequent entries are absorbing layers between sample and
detector.  Thicknesses ``x`` are in cm (or give areal densities ``ad`` in
g/cm2 — note this is g/cm2 here, not the mg/cm2 that `Stack` uses)::

	sp.attenuation_correction(['Fe', ci.Compound('H2O', density=1.0)],
	                          x=[0.1, 0.5])

``sp.geometry_correction(...)`` computes the solid-angle correction for a
sample counted close to the detector, relative to the point-source
geometry of the efficiency calibration, by Monte Carlo integration.  The
dimensions are unitless but must all be in the *same* unit (e.g. all cm),
and ``sample_size`` is the radius for ``shape='circle'`` (the default),
the side length for ``'square'``, or an (x, y) pair for ``'rectangle'``::

	sp.geometry_correction(distance=4, r_det=5, thickness=0.1,
	                       sample_size=2, shape='square')

Plotting, summarizing and saving
--------------------------------

``sp.plot()`` draws the spectrum with its fits (``fit=False`` for the raw
histogram, ``xcalib=False`` for ADC channels on the x-axis).
``sp.summarize()`` prints the counts, decays and activity of each fitted
line.  ``sp.saveas()`` writes the peak table to .csv, .json or .db, the
spectrum itself to .Spe or .Chn (format conversion), or the plot to an
image format::

	sp.saveas('peaks.csv')          # peak table
	sp.saveas('eu_calib_7cm.Chn')   # converted spectrum
	sp.saveas('spectrum.png')       # plot

Calibrations plot with ``cb.plot()`` (all three curves) or individually
(``cb.plot_engcal()``, ``cb.plot_rescal()``, ``cb.plot_effcal()``).
