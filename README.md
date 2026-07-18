# Curie

Curie is a Python toolkit for the analysis of experimental nuclear data, with
primary applications in (gamma-ray) activation analysis and the
charged-particle stacked-target activation technique.  Its name is inspired by
Marie Curie, who pioneered the study of radioactivity.  Alongside the analysis
tools, it provides access to nuclear structure, decay, and reaction databases,
and to atomic properties such as photon attenuation coefficients and
charged-particle stopping powers.

```python
>>> import curie as ci
>>> ci.Isotope('225RA').half_life('d')
14.9
>>> ci.Reaction('115IN(n,g)').plot(scale='loglog')   # a cross section, straight from the libraries
```

Curie's functionality is provided by a small set of classes.  The table below
maps common tasks to the classes that carry them out and the guide section
that walks through each one.

| Task | Classes | Guide |
|------|---------|-------|
| Fit full-energy peaks in HPGe (high-purity germanium) spectra; energy, resolution and efficiency calibration; extract activities | `Spectrum`, `Calibration` | [Spectroscopy](https://jtmorrell.github.io/curie/usersguide/spectroscopy.html) |
| Solve the Bateman equations; fit production rates and end-of-bombardment activities to counting data | `DecayChain` | [Isotopes & Decay Chains](https://jtmorrell.github.io/curie/usersguide/isotopes.html) |
| Retrieve, interpolate, and flux-average evaluated cross sections (ENDF/B-VIII.1, TENDL-2025, IRDFF-II, IAEA) | `Reaction`, `Library` | [Reactions](https://jtmorrell.github.io/curie/usersguide/reactions.html) |
| Compute charged-particle energy loss and flux distributions through a target stack | `Stack` | [Stopping Power Calculations](https://jtmorrell.github.io/curie/usersguide/stopping.html) |
| Look up decay data: half-lives, gamma intensities, branching ratios, atomic masses | `Isotope` | [Isotopes & Decay Chains](https://jtmorrell.github.io/curie/usersguide/isotopes.html) |
| Stopping powers, ranges, and photon attenuation coefficients for elements and compounds | `Element`, `Compound` | [Stopping Power Calculations](https://jtmorrell.github.io/curie/usersguide/stopping.html) |

New to activation analysis itself?  The
[Beginner's Guide](https://jtmorrell.github.io/curie/beginners_guide.html)
introduces the field from the ground up.  Otherwise, install Curie and take the
[Quickstart](https://jtmorrell.github.io/curie/quickstart.html) tour.
Throughout the [documentation](https://jtmorrell.github.io/curie/) (and in all
of the docstring examples), Curie is imported as `import curie as ci`.

## Installation

Curie is available through the [Python Package Index](https://pypi.org/project/curie/):

```
pip install curie
```

Nuclear data is fetched automatically on first use.  Detailed installation
instructions and troubleshooting can be found in the
[installation guide](https://jtmorrell.github.io/curie/quickinstall.html).

## Citing Curie

If Curie contributes to published work, please cite it, for example:
J. T. Morrell, *Curie: A Python toolkit to aid in the analysis of experimental
nuclear data* (2019–), https://jtmorrell.github.io/curie/.
