.. _reactions_troubleshooting:

=========================
Reactions Troubleshooting
=========================

Most reaction-data problems are one of three things: the reaction was
written in a form the library doesn't use, the automatic library choice
wasn't what you assumed, or the target was natural where the library is
isotopic (or vice versa).

"Reaction ... not found or not unique"
--------------------------------------

**Symptom:** ``ValueError: Reaction 48TI(p,n)48V not found or not
unique.`` for a reaction that certainly exists.

**Cause:** the error covers two opposite situations — *nothing* matched
(not found), or *several* things matched (not unique) — and each library
organizes its reactions in one particular form.  The common cases:

* **No product given to a residual-product library** ("not unique"):
  ``'48TI(p,x)'`` alone matches every one of the ~200 products TENDL
  evaluates for that target.  Specify the product.

* **A channel form the library doesn't use** ("not found"):
  ``'natTI(p,n)48V'`` fails in the IAEA library — its entries are
  indexed by product, as ``(p,x)`` — while ``'90ZR(n,x)89ZR'`` fails in
  ENDF, which evaluates exclusive channels: ``'90ZR(n,2n)'``.

* **The reaction genuinely isn't in that library** — the target isn't
  evaluated, or the product is out of reach.

(Related, but not an error: a residual-product search without an isomer
suffix quietly assumes the ground state — ``'86SR(p,x)86Y'`` means
``86Yg``, with a warning that prints only once.)

**Fix:** ask the library what it has, with the loosest search that
brackets your case::

	>>> lb = ci.Library('tendl_p')
	>>> print(lb.search(target='48TI'))
	['48TI(p,x)44SCg', '48TI(p,x)44SCm1', ..., '48TI(p,x)48Vg', ...]

and pass one of the returned names verbatim to `Reaction`.

Which library am I actually using?
----------------------------------

**Symptom:** results change when a colleague runs the "same" analysis;
plots have a different energy range than expected; a cross section has no
uncertainties even though you thought it should.

**Cause:** ``library='best'`` is a *search order*, not a merge: Curie
takes the first library in the priority list (see
:ref:`reactions_tasks`) that contains a unique match.  Two consequences
are easy to miss.  First, similar-looking reactions can come from
different libraries — ``'226RA(n,2n)'`` resolves to ENDF/B-VII.1, but
the residual-product form ``'226RA(n,x)225RA'`` resolves to TENDL-2015 —
with different grids, values and uncertainty availability.  Second, the
resolution can change over time, as data libraries are updated or the
priority evolves between Curie versions.

**Fix:** ``print(rx.library.name)`` — always, before using a number —
and pass the library explicitly (``ci.Reaction('115IN(n,g)', 'irdff')``)
in any analysis you intend to reproduce or publish.

Natural target or isotopic target?
----------------------------------

**Symptom:** ``'natTI(p,x)48V'`` works but ``'natTI(p,x)48V'`` with
``library='tendl_p'`` raises "not found"; or an isotopic cross section
is a factor of several above the measured production on a natural foil.

**Cause:** the libraries differ.  The IAEA monitor library evaluates
*natural* targets (monitor foils are natural metal); TENDL's
residual-product libraries are *isotopic only*.  And the two aren't
interchangeable: an isotopic cross section overstates production on a
natural target by roughly the inverse of the isotope's abundance, and a
natural cross section understates the yield from an enriched target.

**Fix:** match the target to your actual material.  For a natural target
with only isotopic data available, build the abundance-weighted sum (the
:ref:`reactions_tutorial` shows the ~6-line loop, including why some
isotopes contribute nothing).  For an enriched target with only natural
data, there is no clean inversion — use the isotopic library.
