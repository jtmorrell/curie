.. _isotopes:

=======================
Isotopes & Decay Chains
=======================

Curie provides two classes for working with radioactive decay.  The
`Isotope` class looks up nuclear data for a single isotope: masses,
abundances, half-lives, and the energies and intensities of its emissions
(gammas, betas, alphas, conversion electrons).  The `DecayChain` class
does the bookkeeping of decay itself: starting from a parent isotope it
builds the full chain of decay products, computes every member's activity
as a function of time — during production as well as decay — and can fit
production rates or initial activities to measured data.

.. figure:: ../images/ra225_chain.png
   :width: 66%
   :align: center

   The :sup:`225`\ Ra decay chain during and after production: the
   daughters (:sup:`225`\ Ac and its short-lived descendants) grow in as
   the parent decays.

Workflow
--------

Decay-chain problems come in two directions:

* **Forward** — you know (or assume) how much of an isotope was made, and
  want activities at later times: build the chain with an initial activity
  ``A0`` or a production-rate history ``R``, then call ``dc.activity()``,
  ``dc.decays()`` or ``dc.plot()``.

* **Inverse** — you measured decays (typically peaks in gamma-ray spectra)
  and want the production rate or activity that explains them: load the
  measurements with ``dc.get_counts()`` and fit with ``dc.fit_R()`` (for
  production) or ``dc.fit_A0()`` (for decay-only).

Both directions share one time convention: **t = 0 is the end of
production** (the end of bombardment in an activation experiment), and
all times are in the chain's ``units``.

The :ref:`isotopes_tasks` page covers each task in detail, the
:ref:`isotopes_tutorial` works both directions on real examples, and
:ref:`isotopes_troubleshooting` collects the common pitfalls.

Uses and limitations
--------------------

`Isotope` serves quick lookups (a half-life in sensible units, a table of
gamma lines above some intensity) and provides the decay data that
`Spectrum`, `Calibration` and `DecayChain` use internally.  The decay data
are compiled from NuDat 2.0, ENDF/B-VII.0 and the nuclear wallet cards.

`DecayChain` solves the Bateman equations exactly (see
:ref:`methods_decay_chains`), including chains with branching and
same-half-life members, for one parent isotope at a time with a
piecewise-constant production history.  Production physics is *not*
computed — the production rate is an input, which you can estimate from
cross sections and beam parameters (see :ref:`reactions`) or fit from
measured counts.

.. toctree::
   :maxdepth: 1

   isotopes_tasks
   isotopes_tutorial
   isotopes_troubleshooting
