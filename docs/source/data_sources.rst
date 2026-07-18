.. _data_sources:

============
Data Sources
============

Curie's nuclear data ship as pre-built databases, fetched on first use
from versioned data releases and verified against published SHA256
checksums.  Each database carries a generation stamp; this page records
what the current generation (v2, built July 2026) contains and where
every number comes from.  Connecting to data from an earlier generation
logs a warning naming the ``ci.download(...)`` update command.

.. list-table::
   :header-rows: 1
   :widths: 18 44 38

   * - Database
     - Contents and source
     - Version
   * - ``decay``
     - Nuclear structure and decay data: level energies, half-lives,
       branching ratios, and radiation (gamma, X-ray, electron, alpha,
       beta) energies and intensities from ENSDF, retrieved through the
       IAEA LiveChart of Nuclides interface; atomic masses and
       half-life closures from AME2020/NUBASE2020; fission yields
       (independent and cumulative) from the ENDF/B-VIII.1 MF8
       sublibraries; gamma–gamma coincidence probabilities from
       paceENSDF.
     - ENSDF via LiveChart (mirrored 2026-07); NUBASE2020/AME2020;
       ENDF/B-VIII.1
   * - ``endf``
     - Exclusive-channel neutron cross sections, reconstructed from the
       ENDF/B-VIII.1 neutron sublibrary (resonance reconstruction with
       NJOY2016 RECONR at a 0.5 % linearization tolerance).
     - ENDF/B-VIII.1 (2024)
   * - ``tendl``, ``tendl_n/p/d/a``
     - TENDL exclusive-channel (neutron) and residual-product (n, p, d,
       alpha) cross sections, from the TALYS-computed TENDL-2025 tables.
       Targets are curated to the naturally occurring nuclides and
       states with half-lives of at least one year; natural-element
       targets are abundance-weighted sums of the isotopic tables.  The
       proton library's Nb targets are carried from TENDL-2023 ENDF-6
       files (no public host serves TENDL-2025 proton Nb).
     - TENDL-2025 (TALYS-2.1), CC BY 4.0, from
       https://tendl.imperial.ac.uk
   * - ``irdff``
     - The IRDFF-II neutron metrology (dosimetry) standard, with
       uncertainties, from the IAEA's tabulated distribution.
     - IRDFF-II (2020)
   * - ``iaea``
     - IAEA recommended charged-particle monitor reactions and medical
       isotope production cross sections, with uncertainties.
     - IAEA re-evaluation, site as of 2026-07
   * - ``ziegler``
     - Charged-particle stopping-power parameters (Andersen–Ziegler
       formulation), element densities, and NIST XCOM photon
       mass-attenuation coefficients.
     - carried over unchanged between data generations

Acknowledgements and citations
------------------------------

Curie's decay data are extracted from ENSDF through the IAEA Nuclear
Data Section's *LiveChart of Nuclides* API; the fission-yield and
neutron cross-section data derive from ENDF/B-VIII.1 (National Nuclear
Data Center, Brookhaven National Laboratory); TENDL-2025 is distributed
under CC BY 4.0 by the TALYS team; IRDFF-II and the medical monitor
evaluations are published by the IAEA Nuclear Data Section; coincidence
data derive from paceENSDF.  Users publishing results that rest on a
particular evaluation should cite that evaluation directly — the
references collected in :ref:`methods` — in addition to Curie itself.
