
=================================
Welcome to Curie's documentation!
=================================

Curie is a Python toolkit for the analysis of experimental nuclear data, with
primary applications in (gamma-ray) activation analysis and the
charged-particle stacked-target activation technique.  Its name is inspired by
Marie Curie, who pioneered the study of radioactivity.  Alongside the analysis
tools, it provides access to nuclear structure, decay, and reaction databases,
and to atomic properties such as photon attenuation coefficients and
charged-particle stopping powers.

.. code-block:: python

   >>> import curie as ci
   >>> ci.Isotope('225RA').half_life('d')
   14.9
   >>> ci.Reaction('115IN(n,g)').plot(scale='loglog')   # a cross section, straight from the libraries

Curie's functionality is provided by a small set of classes.  The table
below maps common tasks to the classes that carry them out (linked to the
API reference) and the guide section that walks through each one.

.. list-table::
   :header-rows: 1
   :widths: 52 26 22

   * - Task
     - Classes
     - Guide
   * - Fit full-energy peaks in HPGe (high-purity germanium) spectra; energy, resolution and efficiency calibration; extract activities
     - :class:`~curie.Spectrum`, :class:`~curie.Calibration`
     - :ref:`spectroscopy`
   * - Solve the Bateman equations; fit production rates and end-of-bombardment activities to counting data
     - :class:`~curie.DecayChain`
     - :ref:`isotopes`
   * - Retrieve, interpolate, and flux-average evaluated cross sections (ENDF/B-VIII.1, TENDL-2025, IRDFF-II, IAEA)
     - :class:`~curie.Reaction`, :class:`~curie.Library`
     - :ref:`reactions`
   * - Compute charged-particle energy loss and flux distributions through a target stack
     - :class:`~curie.Stack`
     - :ref:`stopping`
   * - Look up decay data: half-lives, gamma intensities, branching ratios, atomic masses
     - :class:`~curie.Isotope`
     - :ref:`isotopes`
   * - Stopping powers, ranges, and photon attenuation coefficients for elements and compounds
     - :class:`~curie.Element`, :class:`~curie.Compound`
     - :ref:`stopping`

New to activation analysis itself?  The :ref:`beginners_guide` introduces the
field from the ground up.  Otherwise, install Curie (:ref:`quickinstall`) and
take the :ref:`quickstart` tour.  Throughout the documentation (and in all of
the docstring examples), Curie is imported as ``import curie as ci``.

**Citing Curie.**  If Curie contributes to published work, please cite it, for
example: J. T. Morrell, *Curie: A Python toolkit to aid in the analysis of
experimental nuclear data* (2019–), https://jtmorrell.github.io/curie/.

.. only:: html

   This documentation is also available as a single PDF: `Curie.pdf <Curie.pdf>`_.

.. toctree::
   :caption: Contents
   :maxdepth: 1

   quickinstall
   quickstart
   beginners_guide
   usersguide/index
   methods
   api/index
   license
