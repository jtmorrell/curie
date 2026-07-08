
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

**What do you want to do?**

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - I want to...
     - use...
     - see...
   * - Fit peaks in a gamma-ray (HPGe) spectrum and get activities
     - ``Spectrum`` + ``Calibration``
     - :ref:`spectroscopy`
   * - Predict or fit activities and decay chains after an irradiation
     - ``DecayChain`` (+ ``Isotope``)
     - :ref:`isotopes`
   * - Look up reaction cross sections from evaluated libraries
     - ``Reaction`` + ``Library``
     - :ref:`reactions`
   * - Plan a stacked-target irradiation (energies, foils, yields)
     - ``Stack`` + ``Compound`` / ``Element``
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
