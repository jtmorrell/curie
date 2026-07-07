.. _isotopes_tasks:

==============================
Isotope & Decay Chain Tasks
==============================

This page is a task-by-task reference for the `Isotope` and `DecayChain`
classes.  All examples assume ``import curie as ci``.  The underlying
equations are in :ref:`methods_decay_chains`.

Looking up an isotope
---------------------

Isotopes are named by mass number and element symbol, in either order::

	ip = ci.Isotope('60CO')
	ip = ci.Isotope('Co-60')    # equivalent

Isomeric (metastable) states are appended to the name: ``m`` or ``m1`` for
the first isomer, ``m2`` for the second, ``g`` for the ground state
(assumed if nothing is given)::

	ip = ci.Isotope('115INm')   # 115In first isomer
	ip = ci.Isotope('Hf-178m2') # 178Hf second isomer (the 31 y state)

Structure properties are attributes: ``ip.mass`` (amu), ``ip.abundance``
(percent), ``ip.E_level`` (MeV), ``ip.J_pi``, ``ip.Delta`` (mass excess,
MeV), ``ip.TeX`` (a LaTeX-formatted name for plot labels).

Half-lives and decay constants
------------------------------

``half_life()`` and ``decay_const()`` take a units argument and can return
uncertainties::

	ip = ci.Isotope('60CO')
	print(ip.half_life('y'))              # 5.271...
	print(ip.half_life('y', unc=True))    # (value, uncertainty)

``ip.optimum_units()`` picks the most readable unit for the half-life —
convenient when looping over isotopes spanning seconds to gigayears::

	print(ip.half_life(ip.optimum_units()), ip.optimum_units())

Emission tables
---------------

The decay radiations are returned as pandas DataFrames, with energies in
keV and intensities in percent (per decay).  Each accepts intensity
(``I_lim``) and energy (``E_lim``) filters::

	ip = ci.Isotope('Co-60')
	print(ip.gammas(I_lim=1.0))          # gamma lines above 1%
	print(ip.electrons(CE_only=True))    # conversion electrons
	print(ip.beta_minus())               # beta- endpoint & mean energies
	print(ip.beta_plus())                # beta+ (positron) emissions
	print(ip.alphas(I_lim=1.0))

``gammas()`` also takes ``xrays=True`` to include fluorescence x-rays and
``dE_511`` to exclude lines near the annihilation energy — the same
filters `Spectrum` applies when fitting peaks.

Fission yields
--------------

For fissioning isotopes, independent fission-product yields are
available: ``get_SFY()`` for spontaneous fission, and ``get_NFY(E)`` for
neutron-induced fission.  Unlike everywhere else in Curie, the incident
energy ``E`` here is in *eV*, on the ENDF grid (0.0253 for thermal, 5E5,
2E6 or 14E6)::

	ip = ci.Isotope('235U')
	print(ip.get_NFY(E=0.0253))    # thermal fission yields

Dose rate
---------

``dose_rate()`` estimates the dose rate from a point source of the
isotope, with no attenuation between source and receptor, for a given
activity (Bq) and distance (cm)::

	ip = ci.Isotope('Co-60')
	print(ip.dose_rate(activity=3.7E10, units='R/hr'))   # 1 Ci at 30 cm
	print(ip.dose_rate(activity=3.7E10, distance=100.0, units='uSv/hr'))

The result is a dictionary with the contribution of each particle type.
Because nothing — not even air — attenuates the particles in this model,
the charged-particle entries are near-contact estimates: at any realistic
distance, alphas and betas would be stopped by the intervening air, so
treat the ``total`` accordingly and use the ``gammas`` entry for
distance-dose estimates.

Building a decay chain
----------------------

A `DecayChain` needs the parent isotope and the time units used by
everything else in the chain — activities, production histories, count
times::

	dc = ci.DecayChain('225RA', units='d')
	print(dc.isotopes)

Valid units are ``'ns'``, ``'us'``, ``'ms'``, ``'s'``, ``'m'``, ``'h'``,
``'d'``, ``'y'``, ``'ky'``, ``'My'``, ``'Gy'`` (with ``'sec'``, ``'min'``,
``'hr'``, ``'yr'`` accepted as synonyms).  ``dc.isotopes`` lists every
chain member found by following the decay data to stability — useful to
check which isotopes the chain actually contains (note the explicit
``g``/``m`` suffixes).

Production and initial activity
-------------------------------

The chain's starting condition is either an initial activity ``A0`` (in
Bq) for decay-only problems::

	dc = ci.DecayChain('152EU', A0=3.7E4, units='y')

or a production-rate history ``R`` — atoms per second produced, as a
piecewise-constant function of time.  Each row is ``[rate, time]``, where
``time`` is the *end* of the interval (a monotonic timestamp grid)::

	# 9/s until t=0.5 d, then 2/s until 1.5 d, then 5/s until 4.5 d
	dc = ci.DecayChain('225RA', R=[[9, 0.5], [2, 1.5], [5, 4.5]], units='d')

With ``timestamp=False`` the times are instead read as interval
*durations* (``[[9, 0.5], [2, 1.0], [5, 3.0]]`` gives the same history).
``R`` can also be a dict keyed by isotope (if daughters are produced
directly), a DataFrame, or a .csv/.json/.db file.  ``dc.R_avg`` gives the
time-averaged rate — the quantity usually quoted from an irradiation.

Activities and decays
---------------------

With the chain defined, any member's activity (Bq) or number of decays
follows, at times measured from the end of production::

	print(dc.activity('225AC', time=10))            # chain units
	print(dc.decays('225AC', t_start=5, t_stop=5.1))
	dc.plot()                                        # all members vs time
	dc.plot(max_plot=5)                              # only the 5 most active

Feeding in measured counts
--------------------------

For the inverse problem, the chain needs measured decays.  The most
direct route is from fitted spectra: ``get_counts()`` reads the
``decays`` column of each spectrum's peak table, keeps the isotopes that
are in this chain, and converts the count timestamps to decay times using
the end-of-bombardment time you supply::

	dc.get_counts([sp1, sp2], EoB='06/12/2026 13:45:00')

``EoB`` (a datetime or a string in ``'%m/%d/%Y %H:%M:%S'`` format) defines
t = 0; each spectrum's start time and duration are taken from its file
header.  Peak data saved earlier with ``sp.saveas()`` works too
(``peak_data='peaks.csv'`` with filenames in ``spectra``).  Counts can
also be entered by hand as ``[t_start, t_stop, decays, unc_decays]`` in
chain units::

	dc.counts = {'225AC': [[5.0, 5.1, 6E5, 2E4],
	                       [6.0, 6.1, 7E5, 3E4]]}

Fitting production rates or activities
--------------------------------------

``fit_R()`` scales the production-rate history to best match the measured
decays; ``fit_A0()`` does the same for the initial activity.  In both
cases the *shape* of what you provided is preserved — the fit adjusts one
overall multiplier per produced isotope, and returns the isotopes, fitted
values and covariance::

	dc = ci.DecayChain('225RA', R=[[9, 0.5], [2, 1.5], [5, 4.5]], units='d')
	dc.counts = {'225AC': [[5.0, 5.1, 6E5, 2E4]]}
	isotopes, R_fit, cov = dc.fit_R()

After the fit, ``dc.R``, ``dc.R_avg`` and all activities reflect the
fitted scale, and ``dc.plot()`` shows the measurements against the fitted
curves.  Points with large uncertainties or few counts are excluded by
the ``max_error`` and ``min_counts`` arguments (40% and 1 by default).
When the counts came from ``get_counts()``, the fit accounts for shared
(correlated) uncertainties between measurements — the gamma intensities
and the efficiency calibration — as described in
:ref:`methods_decay_chains`.
