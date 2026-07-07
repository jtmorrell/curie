.. _isotopes_tasks:

==============================
Isotope & Decay Chain Tasks
==============================

This page is a task-by-task reference for the `Isotope` and `DecayChain`
classes.  All examples assume ``import curie as ci``.  The underlying
equations are in :ref:`methods_decay_chains`.

Looking up an isotope
---------------------

Isotope names have three parts: a mass number, an element symbol, and an
optional state suffix.  Two written forms are accepted::

	ip = ci.Isotope('60CO')     # compact:    mass + SYMBOL (all caps)
	ip = ci.Isotope('Co-60')    # hyphenated: Symbol-mass (any case)

In the **compact** form the element symbol must be entirely upper-case —
the parser reads the capital letters as the symbol, so ``'60Co'`` or
``'60co'`` raise errors rather than finding cobalt.  The **hyphenated**
form is case-insensitive (``'co-60'`` works).  When in doubt, hyphenate.

The state suffix, always lower-case, selects the nuclear state: ``g`` for
the ground state (assumed when no suffix is given), ``m`` or ``m1`` for
the first isomer (metastable excited state), ``m2`` for the second, and
so on::

	ip = ci.Isotope('115INm')   # 115In first isomer
	ip = ci.Isotope('Hf-178m2') # 178Hf second isomer (the 31 y state)

Whatever form you write, Curie normalizes it to one canonical name —
``ip.name``, in compact form with an explicit state — and that canonical
form is what appears everywhere else (``DecayChain.isotopes``, peak
tables, count data):

================  ================  ======================================
You write         ``ip.name``       Meaning
================  ================  ======================================
``'60CO'``        ``60COg``         :sup:`60`\ Co ground state
``'co-60'``       ``60COg``         the same
``'115INm'``      ``115INm1``       :sup:`115`\ In, first isomer
``'Eu-152m2'``    ``152EUm2``       :sup:`152`\ Eu, second isomer
``'99TCg'``       ``99TCg``         :sup:`99`\ Tc ground state, explicit
``'n'``           ``1ng``           a free neutron
================  ================  ======================================

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

``gammas()`` also takes ``xrays=True`` to include fluorescence x-rays,
and ``dE_511`` (a width, in keV) to drop lines within that many keV of
the 511 keV annihilation peak — the same filters `Spectrum` applies when
fitting peaks.

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
This model applies no attenuation, not even from air, so the
charged-particle entries are near-contact estimates; at any realistic
distance, alphas and betas are stopped by the air in between.  Use the
``gammas`` entry for distance-dose estimates, and treat ``total``
accordingly.

Building a decay chain
----------------------

A `DecayChain` needs two things at minimum: the parent isotope, and the
time units that every number in the chain — production histories, count
intervals, the times passed to ``activity()`` — will be interpreted in::

	dc = ci.DecayChain('225RA', units='d')
	print(dc.isotopes)

Valid units are ``'ns'``, ``'us'``, ``'ms'``, ``'s'``, ``'m'``, ``'h'``,
``'d'``, ``'y'``, ``'ky'``, ``'My'``, ``'Gy'`` (with ``'sec'``, ``'min'``,
``'hr'``, ``'yr'`` accepted as synonyms).

At construction, Curie follows every decay mode in the decay data — alpha,
beta, electron capture, isomeric transition, even spontaneous fission —
from the parent down to stable nuclei, and ``dc.isotopes`` lists what it
found, in canonical names with explicit ``g``/``m`` suffixes.  Always
worth printing when a chain misbehaves.  The parent can itself be an
isomer, in which case the chain proceeds through its decay modes::

	>>> dc = ci.DecayChain('178HFm2', units='y')
	>>> print(dc.isotopes)
	['178HFm2', '178HFm1', '178HFg']

The chain does *no* production physics: what gets made, and how fast, is
an input — the subject of the next section.

Production and initial activity
-------------------------------

Beyond the parent and units, the chain needs a starting condition — where
the atoms come from.  There are two, which correspond to the two workflow
directions on the :ref:`isotopes` page:

**Initial activity** ``A0``, for a sample that already exists and simply
decays.  A float is the parent's activity in Bq at t = 0::

	dc = ci.DecayChain('152EU', A0=3.7E4, units='y')

A dict sets several members at once — for example a sample that already
contains some of the daughter::

	dc = ci.DecayChain('99MO', A0={'99MO': 3.7E4, '99TCm': 1.0E3}, units='h')

**Production rate** ``R``, for a sample being made (an irradiation).
``R`` is the number of atoms produced per second, as a piecewise-constant
history: each row is ``[rate, time]``, where ``time`` is the *end* of
that rate's interval::

	# 9/s until t=0.5 d, then 2/s until 1.5 d, then 5/s until 4.5 d
	dc = ci.DecayChain('225RA', R=[[9, 0.5], [2, 1.5], [5, 4.5]], units='d')

Because the rate is atoms per second, activities come out in Bq — a chain
member at saturation has an activity (decays/s) equal to its production
rate (atoms/s).  With ``timestamp=False`` the times are read as interval
*durations* instead (``[[9, 0.5], [2, 1.0], [5, 3.0]]`` is the same
history).

Note the two clocks: the times inside ``R`` run over the irradiation
itself, from 0 (beam on) to the last entry (beam off).  Once the chain is
built, its clock re-zeros — **t = 0 is the end of production**, and
production history appears at negative times (as in the figure on the
:ref:`isotopes` page).

Like ``A0``, ``R`` accepts a dict keyed by isotope, for daughters that
are also produced directly by the reaction (a common situation — e.g.
:sup:`99m`\ Tc made directly alongside its :sup:`99`\ Mo parent)::

	dc = ci.DecayChain('99MO', R={'99MO': [[10, 24]], '99TCm': [[2, 24]]}, units='h')

All isotopes in the dict must share the same time grid.  ``R`` can also
be a DataFrame or a .csv/.json/.db file with columns 'isotope', 'R' and
'time'.  ``dc.R_avg`` gives the time-averaged rate for each produced
isotope — the quantity usually quoted from an irradiation.

Activities and decays
---------------------

With the chain defined, any member's activity (Bq) or number of decays
follows, at times measured from the end of production::

	print(dc.activity('225AC', time=10))            # chain units
	print(dc.decays('225AC', t_start=5, t_stop=5.1))
	dc.plot()                                        # all members vs time
	dc.plot(max_plot=5)                              # only the first 5 chain members

Feeding in measured decays
--------------------------

For the inverse problem, the chain needs measured decays.  A note on
words: in a decay chain, "counts" means the number of *nuclear decays* in
a counting interval — what the spectrum's peak table calls ``decays``,
not its ``counts`` (net peak areas).  The most direct route is from
fitted spectra: ``get_counts()`` reads the ``decays`` column of each
spectrum's peak table, keeps the isotopes that are in this chain, and
converts the count timestamps to decay times using the
end-of-bombardment time you supply::

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
overall multiplier per produced isotope, and returns the isotopes, the
fitted quantities (the time-averaged rate for ``fit_R``, the initial
activity for ``fit_A0``) and their covariance::

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
