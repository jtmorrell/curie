"""
Complete stacked-target cross-section analysis with Curie
=========================================================

A worked, end-to-end example of the charged-particle stacked-target
activation technique -- the produce -> decay -> detect chain from the
Beginner's Guide (https://jtmorrell.github.io/curie/beginners_guide.html).

It uses the geometry of a real natLa(p,x) measurement made at the LBNL
88-Inch Cyclotron in September 2017 (J. T. Morrell et al.,
arXiv:1907.04431).  The raw HPGe spectra, calibrations, and full published
analysis live in the companion repository
https://github.com/jtmorrell/LaCe_Sep2017 -- which used an earlier code,
npat; Curie is its successor.

What this script demonstrates
-----------------------------
  1. Build the foil stack and transport a 57 MeV proton beam through it
     with ``ci.Stack`` -- giving the proton energy distribution in each
     of the ten lanthanum target foils.
  2. Flux-average an evaluated excitation function over each target foil
     with ``ci.Reaction.average`` -- the effective cross section that foil
     probes (a foil spans a range of energies, so the flux-average, not
     the cross section at the mean energy, is the right value; see the
     Stopping-power Worked Examples).
  3. Print those alongside the published measured cross sections.

The *measurement* half of the loop -- fitting the counted spectra with
``ci.Spectrum`` + ``ci.Calibration`` into activities, normalising the beam
current and energy with the copper monitor foils, and inverting the
measured decays to a production rate with ``ci.DecayChain.fit_R`` -- is
walked through in the Spectroscopy and Isotopes tutorials and carried out
in full in the LaCe_Sep2017 repository.  Here we simply show the published
results for comparison.

Note: this plain transport does not reproduce the published energies
exactly -- the full analysis tunes the effective degrader density against
the copper monitor reactions, which shifts the deep-stack energies down by
a few MeV.  The point here is the workflow, not an exact reproduction.

Run with::

    python stacked_target_analysis.py

Needs only ``pip install curie``; the TENDL-2015 proton library is fetched
on first use.
"""
import curie as ci

# ---------------------------------------------------------------------------
# 1. The foil stack, in beam order.
#
# A ~57 MeV proton beam enters through a stainless-steel window (SS3),
# then passes through ten natural-lanthanum target foils (La01-La10), each
# followed by an aluminium catcher (Al) and a copper beam-monitor foil
# (Cu), with aluminium degraders (E..) stepping the energy down between
# groups.  Real (name, compound, density [g/cm^3], thickness [mm]) from the
# LaCe_Sep2017 dataset.  Curie knows SS_316, Kapton, Silicone and Air as
# preset compounds and La/Al/Cu as elements, so nothing custom is needed.
# ---------------------------------------------------------------------------
_foils = [
    ('SS3',    'SS_316',    7.729, 0.13),
    ('La01',   'La',        5.306, 0.0275),
    ('Al01',   'Al',        2.436, 0.027),
    ('Cu01',   'Cu',        7.631, 0.029),
    ('E1',     'Al',        2.698, 0.254),
    ('La02',   'La',        5.595, 0.0278),
    ('Al02',   'Al',        2.399, 0.0278),
    ('Cu02',   'Cu',        7.587, 0.0293),
    ('E2',     'Al',        2.698, 0.254),
    ('La03',   'La',        4.799, 0.0315),
    ('Al03',   'Al',         2.48, 0.027),
    ('Cu03',   'Cu',        7.175, 0.031),
    ('E3',     'Al',        2.698, 0.254),
    ('La04',   'La',        5.192, 0.0288),
    ('Al04',   'Al',        2.473, 0.027),
    ('Cu04',   'Cu',        7.094, 0.0317),
    ('E4',     'Al',        2.698, 0.254),
    ('La05',   'La',         5.58, 0.027),
    ('Al05',   'Al',        2.459, 0.027),
    ('Cu05',   'Cu',        7.154, 0.0313),
    ('E5',     'Al',        2.698, 0.254),
    ('La06',   'La',        5.507, 0.026),
    ('Al06',   'Al',        2.396, 0.0278),
    ('Cu06',   'Cu',        7.169, 0.031),
    ('E6+E7',  'Al',        2.698, 0.508),
    ('La07',   'La',        5.507, 0.0258),
    ('Al07',   'Al',        2.434, 0.0273),
    ('Cu07',   'Cu',        7.227, 0.031),
    ('E8+E9',  'Al',        2.698, 0.508),
    ('La08',   'La',        5.528, 0.0283),
    ('Al08',   'Al',        2.463, 0.0273),
    ('Cu08',   'Cu',        6.926, 0.032),
    ('E10+E11', 'Al',        2.698, 0.508),
    ('La09',   'La',        4.729, 0.0268),
    ('Al09',   'Al',        2.419, 0.0275),
    ('Cu09',   'Cu',        7.162, 0.031),
    ('E12+E13', 'Al',        2.698, 0.508),
    ('La10',   'La',        5.806, 0.0278),
    ('Al10',   'Al',        2.493, 0.027),
    ('Cu10',   'Cu',        7.259, 0.031),
    ('SS4',    'SS_316',    7.789, 0.13),
]
stack = [{'name': n, 'compound': cm, 'density': d, 'thickness': t}
         for (n, cm, d, t) in _foils]

# Transport the beam.  E0/dE0 are the incident energy and 1-sigma spread
# (MeV); N is the number of Monte-Carlo protons.
st = ci.Stack(stack, E0=57.0, dE0=0.4, particle='p', N=20000)

# ---------------------------------------------------------------------------
# 2. Effective cross section in each target foil.
#
# Natural lanthanum is 99.9% 139La, so we take 139La(p,x)135La from the
# TENDL-2015 proton library and flux-average it over each La foil's proton
# energy distribution (from Stack.get_flux).
# ---------------------------------------------------------------------------
rx = ci.Reaction('139LA(p,x)135LAg', 'tendl_p')

# Published measured cross sections (arXiv:1907.04431): foil -> (E, sigma, unc) in MeV, mb.
measured = {
    'La01': (56.05, 120.71, 8.51), 'La02': (54.39, 108.45, 7.49),
    'La04': (50.96, 68.87, 4.64),  'La06': (47.36, 37.76, 3.09),
    'La07': (44.70, 26.12, 1.43),  'La08': (41.88, 8.40, 0.48),
}

print('  natLa(p,x)135La  --  Curie transport + TENDL vs. published measurement')
print('  foil   Curie E   <sigma>       measured E   measured sigma')
print('  ----   -------   --------       ----------   --------------')
for n in range(1, 11):
    name = 'La%02d' % n
    row = st.stack[st.stack['name'] == name].iloc[0]
    eng, flux = st.get_flux(name)          # this foil's energy grid and flux
    sig = rx.average(eng, flux)            # flux-averaged cross section, mb
    m = measured.get(name)
    meas = ('%9.2f   %6.1f +/- %4.1f' % m) if m else ''
    print('  %-5s  %6.2f    %7.1f       %s' % (name, row['mu_E'], sig, meas))

# ---------------------------------------------------------------------------
# 3. Closing the loop (what the full analysis does with the spectra):
#
#   * fit the counted HPGe spectra            -> ci.Spectrum + ci.Calibration
#   * turn peak areas into activities         -> the peaks table / summarize()
#   * normalise the beam with the Cu monitors -> ci.Reaction on natCu(p,x),
#                                                e.g. natCu(p,x)62ZN from IAEA
#   * invert measured decays to a rate        -> ci.DecayChain.get_counts +
#                                                fit_R
#   * divide out beam and target              -> the measured cross section
#
# See the Spectroscopy and Isotopes tutorials for each of these steps, and
# the LaCe_Sep2017 repository for the complete analysis on the real spectra.
# ---------------------------------------------------------------------------
