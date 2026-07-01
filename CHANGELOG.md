# Changelog

Notable changes to curie are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.35] - 2026-07-01

Crash-fix (hotfix) release. Two fixes change numerical results; they are called
out below. All other numbers are unchanged from 0.0.34.

### Fixed
- `Spectrum(..., cb=...)` crashed for `Calibration` objects (mis-parenthesized
  type check).
- `Spectrum.saveas('*.Chn')` crashed for any spectrum with a real sample
  description; description and calibration fields are now padded to the fixed
  widths the reader expects.
- `DecayChain.activity()`/`decays()` raised `KeyError` for stable chain
  members; they now return 0.
- Exactly equal decay constants in a chain produced inf/nan activities
  (`sign(0)=0` defeated the degeneracy guard).
- `Isotope.dose_rate(units='Sv/...')` raised `KeyError`, and the sievert
  weighting was discarded even when it evaluated; alpha dose is now weighted
  by w_R=20.
- `DecayChain.get_counts` raised `NameError` (Python 2 `unicode` reference)
  for non-string start times.
- The `Compound` `.db` round-trip was broken end-to-end (writer and readers
  disagreed on the table name; the compound name was interpolated unquoted).
- A failed `ci.download()` could truncate an existing database to zero bytes,
  and the zero-byte residue was then treated as installed. Downloads now write
  to a temporary file with an atomic rename.
- A missing *proton* reaction with `library='best'` died with a bare
  `IndexError` instead of a clear `ValueError`.
- `DecayChain('238U')` crashed on database records with no atomic mass (71
  exotic nuclides reached through spontaneous-fission daughters); the mass is
  now reconstructed from the mass excess. A missing half-life is treated as
  stable with a visible warning.
- A failed peak fit is now reported with a WARNING naming the isotopes and
  energies instead of being dropped silently, and `fit_peaks` no longer
  crashes when every fit fails.
- Clear error messages for: isotope names absent from the decay database
  (e.g. `Isotope('natFe')` — use `Element`), elements beyond Z=92 (no
  stopping-power/attenuation data), and stack foils whose areal density
  cannot be determined.
- Deprecated `fillna(method='ffill')` calls (pandas 3.x compatibility).
- Four broken documentation examples.

### Changed — numerical results
- **`Compound.range()`** now integrates from 1 keV instead of 1 MeV, matching
  `Element.range()`. Low-energy ranges change by up to ~36% (at 2 MeV);
  the change at 60 MeV is ~0.1%. The old values were wrong.
- **`Library.retrieve`** (and therefore `Reaction`) now sorts energy grids on
  load and drops exact duplicate rows. Several `iaea_monitors` reactions
  (e.g. natFe(p,x)51Cr, 98Mo(n,g)99Mo) store points out of order, which
  corrupted `integrate`/`average`. Results for those reactions change; the
  old values were wrong.

### Changed — packaging
- All nuclear-data downloads (`ci.download()` and the install hook) now fetch
  from the immutable `data-v1` GitHub release (SHA256s in
  `data_manifest.toml`) instead of Dropbox.
- `setup.py` now declares `install_requires` (numpy, scipy, pandas,
  matplotlib) and `python_requires>=3.9`.

## [0.0.34] - 2026-06-08

### Changed
- `Reaction.average` and `Reaction.integrate` now use bin-center midpoint
  quadrature instead of trapezoidal integration, removing a foil-thickness
  dependence in flux-averaged cross sections when the flux spectrum contains
  endpoint spikes (e.g. post-stop residence pile-up from over-stopping foils
  in `Stack`). Docstring example values shift by ~1%.

### Added
- Public test suite (`tests/`, run with `python -m pytest`) and a CI workflow
  covering Python 3.9-3.13.
- `data_manifest.toml`: size and SHA256 records for the shipped nuclear-data
  databases, matching the `data-v1` release assets.

## Earlier releases

Releases up to 0.0.33 predate this changelog.
