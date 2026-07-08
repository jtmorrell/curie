
=================================
Welcome to Curie's documentation!
=================================

Curie is a Python toolkit to aid in the analysis of experimental nuclear data.  Its name is inspired by Marie Curie, who pioneered the study of radioactivity.

The primary application for Curie is (gamma-ray) activation analysis, with specific utilities developed for the charged-particle stacked-target activation technique.
Curie also comes with access to a number of nuclear structure and nuclear reaction databases.  It also has methods for accessing atomic properties,
such as attenuation coefficients and charged particle stopping powers.

Curie's features are primarily class-based.  Here are a few examples:

* Spectrum - Peak fitting for HPGe (high-purity germanium) detector data
* Calibration - Energy, resolution & efficiency calibration tool (for HPGe detectors)
* Element / Compound - Stopping powers, ranges & photon attenuation coefficients
* Stack - Stacked-target energy loss characterization
* DecayChain - General purpose Bateman equation solver
* Isotope - Isotopic mass and decay data
* Reaction - Cross section vs. energy for a single reaction (interpolate, flux-average, plot)
* Library - Search which evaluated libraries carry a given reaction

To get started, visit the :ref:`quickinstall` guide and then
:ref:`getting_started`, which maps out the classes and gives a few
examples to run.  New to activation analysis itself?  The
:ref:`beginners_guide` introduces the field from the ground up.
Throughout the documentation (and in all of the docstring examples),
Curie is imported as::

	import curie as ci

.. only:: html

   This documentation is also available as a single PDF: `Curie.pdf <Curie.pdf>`_.

.. toctree::
   :caption: Contents
   :maxdepth: 1

   quickinstall
   usersguide/index
   methods
   api/index
   beginners_guide
   license
