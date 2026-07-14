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

Tuning the fit
--------------

The defaults are chosen for typical activation spectra; these are the
adjustments that real analyses most often need.  In every case, start from
the fit's own summary line (next section) — it names each filter and how
many lines it dropped, so you can see which knob is in play before turning
it.

**Short or weak counts** — a low-activity sample or a short count leaves
marginal peaks below the ``SNR_min`` threshold (default 4).  Lower it to
admit them (``SNR_min=2.0``), accepting larger uncertainties on the
marginal lines; the reported uncertainties remain honest.

**Busy, many-isotope spectra** — a foil with dozens of activation products
produces hundreds of candidate lines and crowded multiplets.  Raise
``I_min`` (e.g. ``0.5`` for only the strongest branches) and ``SNR_min``
(e.g. ``6``) to fit only the quantifiable lines, and raise ``multi_max``
if wide multiplets are being truncated.  Expect — and read — the
identical-gamma warnings: when two isotopes share a line energy, the
intensity split between them is genuinely ambiguous, and ``ident_idx``
controls how close (in channels) two lines must be to be treated as one.

**Low-energy work** — x-ray and low-gamma lines need ``xrays=True`` and a
lower ``E_min``; the efficiency calibration should then use the
7-parameter model, which the calibration selects automatically when its
spectra include x-rays (see :ref:`methods_calibration`).

**Distorted peak shapes** — strong low-energy tailing or stepped
backgrounds are shape problems, not selection problems; see the
:ref:`spectroscopy_troubleshooting` entries on bad fits
(``skew_fit``/``step_fit``, ``bg``, bounds).

Reading the fit log and diagnostics
-----------------------------------

Every fit announces what it did on the console, in messages of the form
``[LEVEL] Class.method: message``.  The one line to read after any
``fit_peaks()`` is its ``INFO`` summary::

	[INFO] Spectrum(eu_calib_7cm.Spe).fit_peaks: fit 44 peaks in 33 multiplets
	from 2 isotopes; dropped 127 candidates (5 SNR<4.0, 122 intensity<0.05%);
	2 multiplets with chi2/dof>10; 3 peaks with parameters at fit bounds
	(see sp.diagnostics)

Nothing is silently discarded: every candidate line that was not fit is in
one of those counts, with its filter named.  ``WARNING`` messages mark
things that can affect results — a failed multiplet, identical gammas fit
with shared bounds, a calibration evaluated beyond its fitted range.  The
per-item detail behind each summary count is logged at ``DEBUG``::

	ci.set_log_level('DEBUG')   # show every dropped line with its reason
	ci.quiet_warnings()         # errors only
	ci.log_to('curie.log')      # also write the session's messages to a file

The same accounting is available as a table.  ``sp.diagnostics`` has one
row per *attempted* multiplet — including failed ones — with the reduced
chi-square, the model, the uncertainty scale factor, and a greppable
``flags`` column (``at_bound:<param>``, ``chi2_high``, ``fit_failed``);
the ``message`` column holds the full text of everything reported about
that fit.  To pull out the fits that deserve a second look::

	d = sp.diagnostics
	print(d[d['flags'] != ''])

The fitted parameter sets themselves are in ``sp.fits`` (one entry per
multiplet), and ``sp.plot()`` marks any failed multiplet by crosshatching
the unfitted counts in red.

`Calibration` and `DecayChain` follow the same pattern: ``cb.diagnostics``
and ``dc.diagnostics`` summarize their fits, and the calibration point
tables ``cb.engcal_data``, ``cb.rescal_data`` and ``cb.effcal_data`` show
every measured point with a ``used`` flag and the ``reason`` any point was
rejected — rejected points also stay visible as grey open markers on the
calibration plots.  The conventions — message levels, the flags
vocabulary, and how uncertainty scale factors are applied — are defined in
:ref:`methods_reporting`.

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

Choosing the calibration models
-------------------------------

Like the peak fit, ``calibrate()`` is configured through a ``fit_config``
dictionary (``cb.fit_config = {...}`` or keyword arguments, which merge
and persist).  Each calibration has a selectable model — by default the
form of the current calibration is kept, so nothing changes unless you
ask:

=================== =============================================================
Key                 Models
=================== =============================================================
``engcal_model``    ``'linear'``, ``'quadratic'``, ``'cubic'``
``rescal_model``    ``'sqrt'``, ``'linear'``, ``'sqrt_quad'``
``effcal_model``    ``'vidmar'`` (5/7-parameter automatic, the default),
                    ``'vidmar-5'``, ``'vidmar-7'``, ``'loglog'`` (order set by
                    ``effcal_order``, default 4)
=================== =============================================================

For the functional forms and when each is appropriate, see
:ref:`methods_calibration`.  The practical guidance for the efficiency
model: **stay with Vidmar unless it visibly misfits your detector.**  The
semi-empirical form extrapolates smoothly beyond the calibration points;
the log-log polynomial often fits tighter *within* them but diverges
rapidly outside — Curie stores the fitted energy range and warns the first
time an efficiency is evaluated beyond it::

	cb.calibrate([sp], sources=srcs, effcal_model='loglog', effcal_order=5)

The point-selection thresholds are also configurable:
``engcal_max_error``/``rescal_max_error``/``effcal_max_error`` set the
pre-fit uncertainty cuts (defaults 25%/33%/33%), and ``outlier_sigma``
(default 3.16) sets the post-fit residual clip.  The fitted models are
saved in the calibration .json and restored on load; files from older
versions of Curie load unchanged.

Extending the efficiency calibration with known points
------------------------------------------------------

``calibrate(..., eff_points=...)`` appends user-supplied efficiency points
to the measured ones before the efficiency fit — a DataFrame (or list of
dicts) with columns ``energy`` (keV), ``efficiency``, ``unc_efficiency``
and optionally ``isotope``.  Two situations where this is the right tool:

**Covering an energy range your sources don't reach** — e.g. your analysis
needs efficiencies at 2-3 MeV but the calibration sources stop at
1.4 MeV.  Add points from a source measured on another occasion, or from a
detector simulation, and the fit (and its stored energy range) extends to
cover them::

	ep = [{'energy':2000.0, 'efficiency':3.1E-3, 'unc_efficiency':3E-4,
	       'isotope':'56CO'}]
	cb.calibrate([sp], sources=srcs, eff_points=ep)

**Merging counting geometries** — points measured at another distance,
scaled by a solid-angle ratio, can anchor a shared curve shape.  The
user-supplied points are treated as independent measurements (no shared
intensity or source-activity uncertainty), announced in the console, and
carried in ``cb.effcal_data`` with the rest.

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
