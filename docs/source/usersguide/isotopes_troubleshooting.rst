.. _isotopes_troubleshooting:

======================================
Isotope & Decay Chain Troubleshooting
======================================

The failure modes on this page share a theme: `DecayChain` trusts the
names, times and units you give it, and small mismatches — an isomer
suffix, an hour of daylight-saving time — produce quietly wrong answers
rather than errors.

Isotope not found, or the wrong state of the right isotope
----------------------------------------------------------

**Symptom:** ``ValueError: Isotope 99TCm2 not found in the decay
database`` — or, more insidiously, everything runs but half-lives and
gamma lines belong to a different state than the one in your sample.

**Cause:** the isomer suffix.  Curie's naming convention is
``AAAEL`` + state, where the state is ``g`` (ground), ``m`` or ``m1``
(first isomer), ``m2`` (second isomer).  If no state is given, **the
ground state is assumed**: ``ci.Isotope('99TC')`` is the 211,000-year
ground state, not the 6-hour :sup:`99m`\ Tc used in nuclear medicine.
Nothing warns you, because the ground state is a perfectly valid isotope.

**Fix:** be explicit about isomers (``'99TCm'``), and when a chain
misbehaves, print what it actually contains — the names carry explicit
suffixes::

	>>> dc = ci.DecayChain('99MO', units='h')
	>>> print(dc.isotopes)
	['99MOg', '99TCg', '99TCm1', '99RUg']

This matters doubly for measured counts, which are matched by these
exact names.  ``get_counts()`` keeps only isotopes that are in the chain,
so a peak table whose isotope column says ``99TC`` feeds the ground state
``99TCg`` and contributes nothing to a fit of ``99TCm1`` — silently.
(Hand-entered ``dc.counts`` fails louder: naming an isotope that is not a
radioactive chain member raises ``Cannot assign counts to ...``.)

Everything is shifted: time zero and daylight-saving time
---------------------------------------------------------

**Symptom:** fitted activities or production rates that are consistently
a few percent off (or, for short-lived isotopes, wildly off), with a
good-looking fit.

**Cause:** the decay clock.  All of `DecayChain`'s times are measured
from t = 0, which is the **end of production** — for counts loaded with
``get_counts()``, the ``EoB`` (end-of-bombardment) argument you supply
*is* that zero point, and every spectrum's decay time is computed as
(count start - EoB).  Two
things commonly corrupt it:

* **A wrong or mistyped EoB** (or one in ``'%d/%m/%Y'`` order — the
  format is ``'%m/%d/%Y %H:%M:%S'``) shifts every decay time by the same
  amount, which biases every fitted activity by the decay factor of that
  shift.

* **Daylight-saving time.**  Curie's datetimes are timezone-naive; it
  subtracts wall-clock times.  If the irradiation happened in March and
  the count in July, and your EoB is written in winter (standard) time
  while the acquisition computer stamped summer time, every decay time is
  off by one hour.  A one-hour shift is about a sixth of a
  :sup:`99m`\ Tc half-life — roughly an 11% error in activity — or
  negligible for :sup:`152`\ Eu.  Whether it matters depends entirely on
  your shortest-lived isotope.

**Fix:** record EoB and verify the spectrum's ``start_time`` (printed
from the file header: ``print(sp.start_time)``) on the *same* clock,
ideally one that does not observe daylight-saving shifts (UTC, or your
local standard time year-round).  A quick sanity check is to print the
decay times Curie computed::

	dc.get_counts([sp], EoB='03/12/2026 06:30:00')
	print(dc.counts[['isotope', 'start', 'stop']])

and confirm ``start`` (in chain units) is the decay interval you expect.

"Cannot fit R" — or, which fit do I use?
-----------------------------------------

**Symptom:** ``ValueError: Cannot fit R: R=0.`` from ``fit_R()``, or a
``fit_A0()`` result that makes no sense for an irradiated sample.

**Cause:** the two fits answer different questions, and each needs the
matching starting condition:

* ``fit_R()`` scales a **production-rate history** — it requires the
  chain to have been built with ``R``, and answers "what production rate
  explains my measured decays?"  (The error above means the chain has no
  ``R`` to scale.)

* ``fit_A0()`` scales an **initial activity** — for a sample that was
  simply decaying from t = 0, with no production being modeled.

**Fix:** for activation experiments, build the chain with your best
estimate of the production history (even a single interval,
``R=[[1, t_irr]]``, works — the absolute scale is what's being fit) and
use ``fit_R()``.  For a decaying source with no production, give ``A0``
(any nonzero guess) and use ``fit_A0()``.  In both cases the fitted
quantity is a single scale factor per isotope: the *shape* of what you
provided — the beam's time structure, or the relative activities — is
preserved, so a wrong shape (e.g. ignoring a beam interruption) is not
corrected by the fit.
