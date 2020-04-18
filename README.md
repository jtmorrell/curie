# Curie

Curie is a python toolkit to aid in the analysis of experimental nuclear data.  Its name is inspired by Marie Curie, who developed the theory of radioactivity.

The primary application for Curie is (gamma-ray) activation analysis, with specific utilities developed for the charged-particle stacked-target activation technique.
Curie also comes with access to a number of nuclear structure, and nuclear reactions databases.  It also has methods for accessing atomic properties,
such as attenuation coefficients and charged particle stopping powers.

## Features

Curie's features are primarily class based.  Here are a few examples:

* Spectrum - Peak fitting for HPGe detector data
* Calibration - Energy & efficiency calibration tool (for HPGe detectors)
* Stack - Stacked-target energy loss characterization
* DecayChain - General purpose Bateman equation solver
* Isotope - Isotopic mass and decay data
* Reaction - Cross sections from multiple libraries
* Library - Tool for searching and retrieving cross sections from multiple libraries

More detail can be found in the [User's Guide](https://jtmorrell.github.io/curie/build/html/usersguide/index.html) and the [API](https://jtmorrell.github.io/curie/build/html/api/index.html).

## Quick Install

Curie is available through the [Python Package index](https://pypi.org/), which allows installation using Python's standard command line utility [pip](https://pip.pypa.io/en/stable).  Assuming Python and Pip are installed already, you can install Curie with the command:

```
pip install --user curie
```

or:

```
python -m pip install --user curie
```

Detailed installation instructions and troubleshooting can be found in the Curie [documentation](https://jtmorrell.github.io/curie/build/html/quickinstall.html). 



