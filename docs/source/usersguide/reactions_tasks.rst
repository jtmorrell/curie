.. _reactions_tasks:

===============
Reaction Tasks
===============

This page is a task-by-task reference for the `Reaction` and `Library`
classes.  All examples assume ``import curie as ci``.  The library
descriptions and integration conventions are detailed in
:ref:`methods_reaction_data`.

Getting a reaction
------------------

Reactions are written ``TARGET(incident,outgoing)PRODUCT``.  The target
and product take the same isotope names as everywhere in Curie (see
:ref:`isotopes_tasks`), the incident particle is ``n``, ``p`` or ``d``
(plus ``a``, ``h``, ``g`` in the IAEA library), and the outgoing particle
is a shorthand like ``g``, ``2n``, ``p``, ``a``, ``inl`` (inelastic),
``f`` (fission) — or ``x``, meaning "anything": only the product is
specified::

	rx = ci.Reaction('115IN(n,g)')            # radiative capture
	rx = ci.Reaction('Ra-226(n,2n)Ra-225')
	rx = ci.Reaction('natTI(p,x)48V')          # any route from natTi to 48V

A natural-abundance elemental target is written with the ``nat`` prefix
(``natTI``, ``natCU``); this notation is specific to reaction targets —
it is not a valid `Isotope` name.

The product can be omitted when the outgoing particle determines it
(``'115IN(n,g)'`` is ``'115IN(n,g)116IN'``), but is needed to select an
isomer (``'115IN(n,inl)115INm1'``).  The data are attributes: ``rx.eng``
(MeV), ``rx.xs`` (mb), ``rx.unc_xs`` (mb; zeros if the library provides
no uncertainties), and ``rx.TeX`` for plot labels.

How Curie picks the library
---------------------------

With the default ``library='best'``, Curie tries the libraries in a fixed
priority order and keeps the first one that carries the reaction:

======================  ====================================================
Incident particle       Priority order
======================  ====================================================
neutron                 IRDFF-II → ENDF/B-VII.1 → IAEA → TENDL-2015 →
                        TENDL-2015 (residual product)
proton, deuteron        IAEA → TENDL-2015 (residual product)
alpha, helion, photon   IAEA
======================  ====================================================

The order reflects evaluation care: dosimetry and monitor standards
first, general-purpose evaluations next, all-encompassing theoretical
libraries last.  Check which library you got, and pin it explicitly when
reproducibility matters (in a publication, name the library — 'best' can
resolve differently as libraries are added or updated)::

	rx = ci.Reaction('90ZR(n,2n)')
	print(rx.library.name)          # IRDFF-II
	rx = ci.Reaction('90ZR(n,2n)', 'endf')

Exclusive vs. residual-product libraries
----------------------------------------

The neutron libraries (ENDF, TENDL, IRDFF) are organized by *exclusive
reaction channel* — the outgoing particles are specified, as in
``(n,2n)``.  The residual-product libraries (the IAEA library, and the
TENDL ``tendl_n``/``tendl_p``/``tendl_d`` variants) are instead
organized by what nucleus is produced: the reaction is written ``(p,x)``
and only the product matters, summing over every route to it (the full
library taxonomy is in :ref:`methods_reaction_data`).  This is why
proton reactions are written ``'natTI(p,x)48V'`` rather than
``'48TI(p,n)48V'``.

In the TENDL residual-product libraries every product state is a
separate entry: ``86SR(p,x)86Yg`` and ``86SR(p,x)86Ym1`` are different
reactions.  If you give no isomer suffix, **the ground state is
assumed** (a warning prints, once per library instance).

Searching a library
-------------------

When you don't know the exact reaction name, search with any combination
of target, incident/outgoing particle, and product::

	>>> lb = ci.Library('tendl_p')
	>>> print(lb.search(target='Sr-86', product='Y-86g'))
	['86SR(p,x)86Yg']
	>>> lb = ci.Library('endf')
	>>> print(lb.search(target='226RA', outgoing='2n'))
	['226RA(n,2n)225RA']

Searching by target alone lists everything the library evaluates for that
nucleus — for ENDF that includes totals, elastic scattering and
level-by-level inelastic channels, not just activation products.

Library names are ``'endf'``, ``'tendl'``, ``'irdff'``, ``'iaea'``, and
``'tendl_n'``/``'tendl_p'``/``'tendl_d'`` for the residual-product
libraries.  ``search`` returns reaction names ready to pass to
`Reaction`.

Values on your energy grid
--------------------------

``rx.interpolate(energy)`` evaluates the cross section wherever you need
it, and ``rx.interpolate_unc(energy)`` its uncertainty::

	rx = ci.Reaction('115IN(n,g)', 'irdff')
	print(rx.interpolate([0.5, 1.0, 5.0]))       # mb, at 0.5, 1 and 5 MeV

Interpolation is linear (quadratic for the smooth TENDL curves).  One
convention to know: **outside the library's evaluated energy range the
interpolated cross section is zero** — Curie never extrapolates.  Check
``rx.eng.min()`` and ``rx.eng.max()`` when working near the edges of a
library's grid; the consequences for flux averages are shown in the
:ref:`reactions_tutorial`.

Flux averages and integrals
---------------------------

For an activation measurement, the effective cross section is the
flux-weighted average over the particle spectrum.  Give ``average()``
your energy grid and the flux on that grid::

	eng = np.linspace(10, 30, 50)
	phi = np.exp(-0.5*((eng - 20)/3.5)**2)       # any relative shape
	print(rx.average(eng, phi))
	print(rx.average(eng, phi, unc=True))        # (value, uncertainty)

The flux needs no normalization — only its shape matters for the
average.  ``rx.integrate(eng, phi)`` returns the flux integral
:math:`\int \sigma(E)\,\phi(E)\,dE` instead, for which the flux's
absolute normalization *does* matter (which to use when, and how each
becomes a production rate, is laid out on the :ref:`reactions` overview
page).  Cross-section uncertainties are treated as fully correlated
between energies — see :ref:`methods_reaction_data` for the exact
conventions.

Plotting
--------

``rx.plot()`` shows the cross section over its full grid, with an
uncertainty band when the library provides one.  Overlay reactions or
libraries by passing figure and axes through::

	rx = ci.Reaction('90ZR(n,2n)', 'irdff')
	f, ax = rx.plot(return_plot=True, label='library')
	rx = ci.Reaction('90ZR(n,2n)', 'endf')
	rx.plot(f=f, ax=ax, label='library')

``label`` can be ``'reaction'``, ``'library'`` or ``'both'``; pass
``energy=`` to plot on your own grid; ``scale='loglog'`` (and friends)
sets the axes.
