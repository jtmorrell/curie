.. _stopping_troubleshooting:

==============================
Stopping Power Troubleshooting
==============================

Stack problems usually appear as energies that don't match
expectations.  The three failure modes below cover most cases; for all of
them, the first diagnostic is the same — ``print(st.stack)`` and check
the ``areal_density`` column against what you think is in the beam.

Foil energies are far from expectation
--------------------------------------

**Symptom:** ``mu_E`` values much too low (or the beam exhausts halfway
down the stack) even though the foils are thin; or energies that barely
change through what should be a substantial degrader.

**Cause:** almost always units.  The stack input mixes several, and each
must be right:

* ``thickness`` (``'t'``) is in **mm** — a 25 um foil is ``0.025``, and
  typing ``25`` makes it a 2.5 cm slab.
* ``areal_density`` (``'ad'``) is in **mg/cm2**.
* ``mass`` is in **g** and ``area`` in **cm2**.
* ``density`` is in **g/cm3**.

**Fix:** print ``st.stack`` and inspect the computed ``areal_density``
of each foil — a 25 um Ti foil should be ~11 mg/cm2, a 0.5 mm Al foil
~135 mg/cm2.  If a foil is off by a factor of 10–1000, one of its input
units is wrong.

"WARNING: Beam stopped in foil N"
----------------------------------

**Symptom:** the warning above, foils downstream of foil N with
``mu_E`` near zero, and nonsense fluxes from ``get_flux`` for those
foils.

**Cause:** the beam physically ranges out — the summed thickness of the
stack exceeds the particle range at ``E0``.  (``N`` is the row index
shown by ``print(st.stack)``, counting from 0, so the offending foil is
``st.stack.iloc[N]``.)  Every downstream energy and flux is meaningless
(Curie transports energy loss only; a stopped beam has no energy left to
assign).  This is sometimes intentional — a beam dump or catcher at the
end — and then harmless, as long as the foils you *analyze* sit upstream
of the stopping point.

**Fix:** if the stack *should* pass the beam easily, suspect a
thickness-unit error first (previous section) before concluding it truly
ranged out.  Then compare the range against the summed foil thicknesses
— in matching units::

	>>> print(10*ci.Element('Al').range(30.0))   # x10: range in mm, like 't'
	4.359...

and either raise ``E0``, thin the stack, or — if the stop is intentional
— simply ignore the foils at and beyond the stopping depth.

A foil is missing from the results
----------------------------------

**Symptom:** a foil you defined does not appear in ``st.summarize()``,
``st.plot()`` or ``st.fluxes``; or ``get_flux('Ti-01')`` returns empty
arrays.

**Cause:** foils without a ``'name'`` are deliberately untallied — they
degrade the beam but are treated as passive (degraders, spacers,
catchers).  Only named foils are stored in the results.  And
``get_flux`` matches the name **exactly**: ``'Ti01'`` and ``'Ti-01'``
are different strings.

**Fix:** give every foil you care about a ``'name'`` in the stack
definition, and read the exact spellings back from ``print(st.stack)``
(the ``name`` column) before calling ``get_flux``.  Unnamed foils still
affect the transport — their absence from the results does not mean they
were skipped.
