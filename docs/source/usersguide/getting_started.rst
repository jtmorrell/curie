.. _getting_started:

===============
Getting Started
===============

Once Curie is installed (see :ref:`quickinstall`), everything is reached
through a single import::

	import curie as ci

This page maps out the toolkit — which class does what, and where to read
more — and gives a few short examples to run.

The Curie toolbox
-----------------

Curie's functionality lives in a handful of classes, grouped here by what
you would use them for.  The classes are designed to work together: peaks
fit from a spectrum feed a decay-chain fit, a foil's particle flux feeds a
cross-section average, and so on.

**Fitting gamma-ray spectra.**  `Spectrum` fits the peaks in HPGe detector
data, using a `Calibration` (energy, efficiency and resolution) and
`Isotope` decay data to turn peak areas into activities.  See
:ref:`spectroscopy`.

**Production and decay calculations.**  `Isotope` provides decay data —
half-lives, gamma-rays, dose rates — for a single nuclide, while
`DecayChain` solves the Bateman equations for a whole chain: forward
(predicting activities from a production rate) or backward (fitting a
production rate or initial activity to measured decays, including peaks
read straight from a `Spectrum`).  See :ref:`isotopes`.

**Nuclear reaction data.**  `Reaction` gives a cross section as a function
of energy, with interpolation, flux-averaging and plotting; `Library`
searches the evaluated libraries (ENDF, TENDL, IRDFF, IAEA) for what is
available.  See :ref:`reactions`.

**Stopping powers and stacked targets.**  `Element` and `Compound` compute
charged-particle stopping powers, ranges and photon attenuation
coefficients; `Stack` transports a beam through a stack of foils to find
the particle energy in each — a flux that can be handed straight to
`Reaction.average`.  See :ref:`stopping`.

For the models and formulas behind these methods, see the
:ref:`methods` chapter; for every method and attribute, the :ref:`api`.

A few things to try
-------------------

Fit the peaks in a gamma-ray spectrum of a :sup:`152`\ Eu source (the
``eu_calib_7cm.Spe`` file ships with Curie)::

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']
	sp.plot()                       # the spectrum, with its fitted peaks

Look up the decay data for a radionuclide::

	ip = ci.Isotope('225RA')
	print(ip.half_life('d'))        # 14.9
	print(ip.gammas())              # a table of its decay gamma-rays

Plot an evaluated reaction cross section::

	rx = ci.Reaction('115IN(n,g)')  # neutron capture on 115In
	rx.plot(scale='loglog')

Each links to a fuller, worked walk-through: :ref:`spectroscopy`,
:ref:`isotopes`, :ref:`reactions`.

Example scripts
---------------

Complete, runnable scripts covering each of these areas ship in the
`examples <https://github.com/jtmorrell/curie/tree/master/examples>`_
directory of the Curie repository, alongside the example data files they
use.

.. Example scripts are currently minimal; expand them into fuller worked
   problems (and add a ci.run_demo() entry point) in a future docs pass.
