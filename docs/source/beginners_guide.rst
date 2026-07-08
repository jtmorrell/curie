.. _beginners_guide:

=========================================
A Beginner's Guide to Activation Analysis
=========================================

This chapter is a conceptual introduction to activation analysis — the
field Curie was built to serve — for readers who are new to it.  Where the
:ref:`usersguide` shows *how* to drive Curie and the :ref:`methods` chapter
gives the equations, this guide explains *why* the analysis looks the way
it does: what is being measured, and how each number leads to the next.
Each section ends with a short pointer to the Curie classes that carry out
that step.

.. _bg_what:

What is activation analysis?
============================

The idea behind activation analysis is to learn about matter by making it
briefly radioactive and watching it decay.  A sample is exposed to a beam
or flux of particles — neutrons from a reactor, or charged particles
(protons, deuterons, alphas) from an accelerator.  A fraction of the nuclei
in the sample undergo nuclear reactions and are transformed into new,
often radioactive, nuclei.  Those product nuclei then decay, emitting
radiation — most usefully gamma rays, whose energies are sharp and
characteristic of the emitting isotope.  By detecting that radiation, you
can identify which isotopes were produced and, from how much radiation you
see, *how much* of each.

That single chain of events — **produce, decay, detect** — underlies two
kinds of measurement Curie supports:

* **Quantifying activity.**  Given a sample that has been irradiated, how
  active is each isotope in it?  This is the everyday task of gamma-ray
  spectroscopy: turn a measured spectrum into a table of activities.

* **Measuring cross sections.**  How readily does a particular reaction
  occur, as a function of the beam energy?  This is the goal of the
  charged-particle stacked-target technique, where the measured activities
  are worked backward into the underlying reaction probabilities.

The rest of this guide follows the produce–decay–detect chain forward and
backward: the physics of production and decay, how activity is measured,
how a measured activity becomes a production rate, how charged particles
behave in a target, and finally how all of this is combined to predict
yields and design an experiment.

**In Curie.**  Every class in the toolkit sits somewhere on this chain; the
:ref:`quickstart` maps them out.

.. _bg_production:

Production, decay, and counting
===============================

Three intervals of time structure every activation measurement: the sample
is **irradiated**, then it **cools** (decays) for a while, and then it is
**counted** in a detector.  Following one product isotope through these
intervals gives the whole quantitative framework.

**Production.**  During irradiation the product is created at some rate
:math:`R` (atoms per second) and, being radioactive, decays at the same
time with decay constant :math:`\lambda = \ln 2 / t_{1/2}`.  The number of
product atoms climbs until creation and decay balance.  For a constant
production rate over an irradiation of duration :math:`t_{\mathrm{irr}}`,
the activity at the end of the irradiation — the *end of bombardment*, or
EoB — is

.. math::

   A_{\mathrm{EoB}} = R\,\bigl(1 - e^{-\lambda t_{\mathrm{irr}}}\bigr).

The term in parentheses is the **saturation factor**.  Irradiate for one
half-life and you reach half the maximum possible activity; irradiate for
several half-lives and the activity *saturates* at :math:`A = R` — no
matter how long you keep going, the isotope decays as fast as it is made.
This is the single most important consequence of the half-life for
planning: short-lived products saturate quickly (little is gained by a
long irradiation) while long-lived products build up slowly.

**Decay.**  After the beam is off, the activity simply decays:

.. math::

   A(t) = A_{\mathrm{EoB}}\,e^{-\lambda t},

with :math:`t` measured from EoB.  Time zero for the whole analysis is
conventionally this end-of-bombardment moment.

**Counting.**  A detector does not measure activity directly; over a
counting interval it records the *number of decays* that occur in that
window (times the fraction it manages to detect).  Because the activity is
itself falling during a long count, the number of decays is the integral
of :math:`A(t)` across the counting interval, not simply activity times
time.  Keeping these three intervals — and their clocks — straight is the
bookkeeping that the analysis lives or dies by.

**In Curie.**  `DecayChain` implements exactly this, and generalizes it
from one isotope to a full decay chain (parents feeding daughters) via the
Bateman equations.  See :ref:`isotopes` and, for the equations,
:ref:`methods_decay_chains`.

.. _bg_spectroscopy:

Measuring activity: gamma-ray spectroscopy
==========================================

The most common way to observe the decays is gamma-ray spectroscopy with a
high-purity germanium (HPGe) detector.  Each decaying isotope emits gamma
rays at a set of characteristic energies, each with a known *intensity* —
its emission probability, the fraction of decays that produce that gamma.
The detector sorts detected gammas by energy into a **spectrum** —
a histogram of counts versus energy — in which each isotope appears as
peaks at its own line energies.

To turn the area of a peak into an activity, three calibrations are needed,
and it is worth understanding why each is required:

* **Energy calibration** maps detector channel to gamma energy, so that a
  peak can be matched to the isotope and line that produced it.

* **Resolution calibration** describes how wide the peaks are, which the
  fitting routine needs to separate nearby lines.

* **Efficiency calibration** gives the fraction of gammas emitted at a
  given energy that are actually recorded as full-energy counts.  Without
  it a peak area is only a relative number; with it, the count rate becomes
  an absolute emission rate.  Efficiency falls with distance and varies
  strongly with energy, so it must be measured for the specific detector
  and geometry, using a source of known activity.

Given these, the number of counts in a peak relates to the activity through
the gamma intensity, the efficiency, and the counting-time factors of the
previous section.  Inverting that relation for every clean peak yields the
isotope's activity — the measured quantity that the rest of the analysis
builds on.

**In Curie.**  `Spectrum` fits the peaks and `Calibration` produces and
stores the three calibrations.  See :ref:`spectroscopy`, and
:ref:`methods_peak_fitting` and :ref:`methods_calibration` for the models.

.. _bg_activity_to_rate:

Turning activities into production rates
========================================

Measuring an activity is a means to an end.  What you usually want is the
quantity that was under experimental control: the **production rate**
during irradiation (or, for a decay-only sample, the activity at some
reference time).  This is the produce–decay–detect chain run *backward*.

For a single isotope it is a matter of algebra: rearranging the saturation
relation from :ref:`bg_production`,

.. math::

   R = \frac{A_{\mathrm{EoB}}}{1 - e^{-\lambda t_{\mathrm{irr}}}},

and :math:`A_{\mathrm{EoB}}` itself is obtained by decay-correcting the
activity measured at count time back to EoB.  In practice several
complications enter at once: the isotope of interest may be fed by the
decay of a parent, several spectra taken at different cooling times each
constrain the same production rate, and the beam may have varied during the
irradiation.  The robust way to handle all of this is to *fit*: adjust the
production rate until the decays it predicts, across the whole chain and
all counting intervals, best match the measured ones.

A useful distinction: fitting a **production rate** applies when the sample
was being made during an irradiation, whereas fitting an **initial
activity** applies to a sample that was simply decaying from some reference
time (a calibration source, say).  The two answer different questions and
take different starting information.

**In Curie.**  `DecayChain.get_counts` reads measured decays (including
straight from fitted `Spectrum` peaks), and `fit_R` and `fit_A0` fit a
production rate or an initial activity to them.  The
:ref:`isotopes_tutorial` works a full inverse example.

.. _bg_charged_particle:

Charged-particle interactions in a target
==========================================

So far the beam could have been anything.  Charged-particle beams behave in
a way that shapes the entire stacked-target technique, so they deserve
their own discussion.

Unlike neutrons, which travel until they happen to react, a charged
particle interacts continuously with the electrons of the material it
traverses, losing energy little by little.  The rate of energy loss per
unit path length is the **stopping power**.  Two consequences follow, and
both are central to activation work:

* **The beam energy decreases with depth.**  A particle that enters a foil
  at one energy leaves it at a lower one, and eventually — after a distance
  called its **range** — stops entirely.  Because a reaction's cross
  section depends on energy, the production rate is not uniform through a
  thick target: different depths are effectively irradiated at different
  energies.

* **The beam spreads in energy.**  A foil sees a *range* of energies, not a
  single value: partly because the beam loses energy continuously across
  the foil's own thickness, and partly because any initial energy spread
  grows as the beam slows (slower particles lose energy faster).  Random
  fluctuations in the energy loss (*straggling*) broaden it further still.

These facts are usually a nuisance — but they can be turned into an
advantage.  If a thin foil barely changes the beam energy, it measures a
reaction essentially at a single energy.  Stack many thin foils, with
degraders between them to step the energy down, and one irradiation samples
the reaction at a whole ladder of energies at once.  This is the
**stacked-target** technique, and reading the energy in each foil correctly
is the crux of it.

**In Curie.**  `Element` and `Compound` compute stopping powers and ranges;
`Stack` transports a beam through a stack of foils and reports the energy
distribution in each.  See :ref:`stopping` and :ref:`methods_stopping`.

.. _bg_yields:

Predicting production rates and isotope yields
==============================================

With the pieces in hand, the forward calculation — how much of a product
will an irradiation make? — comes together.  In a single foil, the
production rate is the reaction cross section combined with the beam and
the target:

.. math::

   R = I_p \; n_t \; \langle\sigma\rangle,

where :math:`I_p` is the beam current (particles per second),
:math:`n_t` the number of target atoms per unit area, and
:math:`\langle\sigma\rangle` the cross section *averaged over the energy
distribution of the beam in that foil* — the foil's energy spectrum, which
`Stack.get_flux` supplies.  (Curie also folds in the unit factor converting
mb to cm2; the :ref:`reactions` page gives the fully dimensioned
form.)
That flux-averaging is where the
previous section pays off: for a thin foil the average is essentially the
cross section at the beam energy, but for a thick foil, where the beam
spans a wide energy range, using the cross section at the mean energy alone
can be badly wrong — the average must be taken over the real distribution.

Once :math:`R` is known, the activation equation of :ref:`bg_production`
turns it into an activity at end of bombardment, and the decay and counting
relations turn *that* into the number of decays a detector would record at
any later time.  The chain is now closed: a cross section, a beam, and a
target predict the very counts that a measurement would produce — and,
run the other way, measured counts yield the cross section.

**In Curie.**  `Reaction.average` (and `Reaction.integrate`) combine a
cross section with a beam spectrum — for a foil, ``Stack.get_flux`` supplies
that spectrum — and the result feeds `DecayChain` as a production rate.
See the "Averages vs. integrals" discussion on the :ref:`reactions` page,
and the :ref:`stopping_tutorial` for the thin-versus-thick comparison.

.. _bg_design:

Designing a stacked-target experiment
=====================================

A stacked-target cross-section measurement is planned backward from what it
needs to produce: measurable activities of the product isotopes, at a set
of well-known beam energies, with the beam current under control.  The main
design choices are:

* **Energy coverage.**  The incident beam energy and the stack's degraders
  set the range of energies sampled.  Enough foils are included, with
  degraders sized so that consecutive target foils sit at usefully spaced
  energies across the region of interest.

* **Target foils.**  Each target foil must be thin enough that the beam
  energy is well defined across it (so the measured cross section belongs
  to a narrow energy), yet thick enough to produce enough activity to
  measure.  These pull in opposite directions and are traded off against
  the expected cross section and beam current.

* **Monitor foils.**  Interleaved foils of well-characterized materials
  carry *monitor reactions* — reactions whose cross sections are known to
  high accuracy.  Measuring their activities reconstructs the actual beam
  current and energy at each position in the stack, which is how the
  experiment calibrates itself rather than trusting the nominal beam
  parameters.

* **Cooling and counting schedule.**  The half-lives of the products
  dictate the timing: short-lived isotopes must be counted soon after
  irradiation, long-lived ones may need to cool so that short-lived
  activities decay away first.

* **Range check.**  The beam must actually reach the last foil of interest:
  the material in front of it has to stay within the particle's range at the
  incident energy (degraders or catchers placed further downstream may lie
  beyond the range without harm).

The Curie workflow mirrors the experiment.  *Before* the beam time, define
the foils and use `Stack` to predict the energy in each, then combine
monitor and product cross sections with those fluxes to estimate the
activities you will produce — confirming the design will yield measurable,
well-placed data.  *After* the irradiation, fit the counted spectra with
`Spectrum` and `Calibration`, decay-correct and fit production rates with
`DecayChain`, normalize to the monitor foils, and divide out the beam and
target to recover the cross sections.

**In Curie.**  This ties together every part of the toolkit; the four topic
groups of the :ref:`usersguide` cover the individual steps, and their
troubleshooting pages collect the pitfalls (unit mix-ups, a beam that stops
in the stack, thick-foil energy spread, monitor-reaction choice) that most
often trip up a first experiment.
