.. _stopping_howto:

===========================
Stopping Power How-to Guide
===========================

This page is a task-by-task reference for the `Element`, `Compound` and
`Stack` classes.  All examples assume ``import curie as ci``.  The
stopping-power formulation and the transport algorithm are described in
:ref:`methods_stopping`.

Elements
--------

An `Element` is a natural-abundance element, constructed from its
symbol::

	el = ci.Element('Fe')
	print(el.mass)        # amu, abundance-weighted
	print(el.density)     # g/cm3, at standard conditions
	print(el.abundances)  # table of stable isotopes and abundances

The ``abundances`` table (columns ``isotope``, ``abundance`` in percent)
is the same one used to abundance-weight isotopic cross sections in the
:ref:`reactions_examples`.  The preset ``density`` is used as the default
by the methods below; pass ``density=`` to override it.

Photon attenuation
------------------

``mu()`` and ``mu_en()`` return the mass-attenuation and
mass-energy-absorption coefficients (cm2/g) at a given photon energy (in
keV, as everywhere photons appear), and ``attenuation()`` the fraction of
photons transmitted through a thickness ``x`` (cm)::

	el = ci.Element('Pb')
	print(el.mu(661.7))                  # cm2/g at the 137Cs line
	print(el.attenuation(661.7, x=1.0))  # transmission through 1 cm

These are the same coefficients `Spectrum.attenuation_correction()` uses
for sample self-absorption (see :ref:`spectroscopy_howto`).

Stopping powers and ranges
--------------------------

``S()`` returns the stopping power :math:`-dE/dx` for a charged particle,
at an energy in MeV::

	el = ci.Element('Ti')
	print(el.S(15.0))                 # MeV/cm, at the preset density
	print(el.S(15.0, density=1E-3))   # MeV/(mg/cm2) - mass stopping power

The ``density=1E-3`` idiom returns the *mass* stopping power, which is
what stacked-target work uses (areal densities in mg/cm2).  The particle
is ``'p'`` (default), ``'d'``, ``'t'`` or ``'a'`` — or any element or
isotope name (``'Fe'``, ``'40CA'``) for heavy ions.

``range()`` integrates the stopping power to give the distance a
particle travels before stopping, in cm::

	print(el.range(15.0))             # ~0.087 cm for 15 MeV protons in Ti

Both have plotting companions, ``plot_S()`` and ``plot_range()``, which
accept the shared plotting keywords (overlay with ``f``/``ax``).

Compounds
---------

A `Compound` is defined by elemental weights and a density.  Three ways
to make one::

	cm = ci.Compound('H2O', density=1.0)     # chemical formula
	cm = ci.Compound('Kapton')               # preset (see ci.COMPOUND_LIST)
	cm = ci.Compound('Brass', weights={'Cu':-66, 'Zn':-33}, density=8.5)

Formulas support decimal subscripts (``'C0.5O'``) but not parentheses —
write Ca3(PO4)2 as ``'Ca3P2O8'``.  In a ``weights`` dict, **positive
numbers are atom fractions, negative numbers are mass fractions** (the
brass above is 66% copper *by weight*); either kind is normalized for
you.  Weights can also come from a .csv/.json/.db file describing many
compounds at once.  A `Compound` has the same ``mu``, ``mu_en``,
``attenuation``, ``S``, ``range`` and plotting methods as an `Element`,
with elemental values combined as mass-fraction-weighted sums of the
elemental ones (Bragg additivity — see :ref:`methods_stopping`).

Building a stack
----------------

A stack is an ordered list of foils — first foil hit first — passed to
`Stack` with the beam parameters::

	stack = [{'cm':'Al', 't':0.5,   'name':'Al01'},
	         {'cm':'Ti', 't':0.025, 'name':'Ti01'},
	         {'cm':'Kapton', 't':0.05},
	         {'cm':'Cu', 't':0.025, 'name':'Cu01'}]

	st = ci.Stack(stack, E0=30.0, particle='p')

Each foil needs a compound (``'compound'``, shorthand ``'cm'``) and
enough information to fix its **areal density** in mg/cm2, given
directly (``'areal_density'``/``'ad'``) or computed from: ``'mass'``
(g) with ``'area'`` (cm2), or ``'thickness'`` (mm) with ``'density'``
(g/cm3) — the density coming from the compound definition if not given
per-foil.  Every key has a shorthand, used in the example above:
``cm`` = compound, ``t`` = thickness, ``d`` = density, ``m`` = mass,
``a`` = area, ``ad`` = areal_density, ``nm`` = name.

Foils with a ``'name'`` are tallied; unnamed foils (the Kapton above)
still degrade the beam and still appear in ``st.stack``, but are
excluded from the tallied results (``st.fluxes``, ``get_flux``,
``plot``, ``summarize``, ``saveas``) — the convention for degraders and
catchers.  The stack can equally be a DataFrame or a .csv/.json/.db
file with the same columns, and custom compounds can be supplied with
the ``compounds=`` argument.

Reading the results
-------------------

``st.stack`` summarizes every foil: its areal density, the mean beam
energy ``mu_E`` and the 1-sigma energy width ``sig_E`` (MeV)::

	>>> print(st.stack)
	   name compound  areal_density       mu_E     sig_E
	0  Al01       Al       134.9000  29.015085  0.730443
	1  Ti01       Ti        11.2975  27.935964  0.321319
	...

The full energy distributions are in ``st.fluxes``, plotted with
``st.plot()``, and retrieved per foil with ``get_flux()`` — whose output
feeds directly into a cross-section average::

	rx = ci.Reaction('natTI(p,x)48V')     # a reaction on the foil material
	eng, phi = st.get_flux('Ti01')
	print(rx.average(eng, phi))

``st.summarize()`` prints the energy assignments, and ``st.saveas()``
writes the stack table (and optionally the fluxes) to .csv/.json/.db.

Solver options
--------------

The defaults suit most problems; the parameters, all keyword arguments to
`Stack`: ``dE0`` (1-sigma width of the incident beam energy, default 1%
of ``E0``), ``N`` (number of Monte Carlo particles, default 10000),
``accuracy``/``min_steps``/``max_steps`` (energy-loss stepping control
per foil), and ``dp`` (a density multiplier applied to the whole stack —
useful as a fit parameter when tuning a stack model to measured monitor
data).
