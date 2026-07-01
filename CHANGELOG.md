# Changelog

Notable changes to curie are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
