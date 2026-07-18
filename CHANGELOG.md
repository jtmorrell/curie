# Changelog

Notable changes to curie are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed — nuclear data generation v2
- **All data libraries rebuilt from current evaluations**: ENDF/B-VIII.1
  (was VII.1), TENDL-2025 (was 2015), IAEA medical monitors 2025, and
  decay data from ENSDF via IAEA LiveChart with NUBASE2020/AME2020
  (`Library.name` reports the new versions). Values everywhere reflect
  the newer evaluations; target coverage differs where the evaluations
  themselves changed (for example ENDF/B-VIII.1 provides Cd-115m but no
  Cd-115 ground-state evaluation).
- **TENDL target curation**: the TENDL libraries carry targets that are
  naturally occurring or have half-lives of at least one year (ground
  states and isomeric targets alike, each state judged on its own
  half-life), including long-lived second isomers (178HFm2, 192IRm2)
  and the ten classic long-lived first isomers.
- **Data generation awareness**: each database now carries a generation
  stamp; connecting to data fetched by an earlier curie release logs a
  warning naming the `ci.download(...)` repair, and `ci.download()`
  refreshes outdated files without needing `overwrite=True`.
- `Isotope`: a state with no measured half-life in the decay data now
  reports `stable = False` with an infinite half-life, instead of being
  presented as stable.

### Added
- **Alpha-particle residual-product library**: `ci.Library('tendl_a')`
  (TENDL-2025), with `'best'` for alpha-incident reactions searching
  the IAEA monitors first and then tendl_a_rp — mirroring protons and
  deuterons.
- **Natural-element targets in the TENDL residual-product libraries**:
  `ci.Reaction('natFe(p,x)56CO', 'tendl_p')` now works for ~80 elements
  in all four residual-product libraries (n, p, d and the new alpha
  library). The cross sections are abundance-weighted sums of the
  isotopic tables, computed at data-build time from the same decay-data
  generation; elements whose lightest isotopes TENDL does not provide
  (H, He, Li, Be) are not available. The `natEl` notation matches the
  IAEA monitor and IRDFF libraries.

## [0.2.0] - 2026-07-14

Fitting transparency release: every fit reports a summary of its results
and selections, retains the measurements it rejected, and exposes
fit-quality diagnostics. Zero-configuration results are unchanged,
with one deliberate exception noted under Fixed (the unresolvable-doublet
merge). Console message texts have changed throughout — anything parsing
curie's printed output will need updating; the messages now follow one
documented convention (`[LEVEL] Class.method: message`).

### Added — logging
- **A logging system replaces bare print()**: messages go to stdout as
  `[LEVEL] Class.method: message`, with per-item detail (each dropped line,
  each parameter at a fit bound) at DEBUG and summary accounting at INFO.
  `ci.set_log_level('DEBUG')` shows everything, `ci.quiet_warnings()`
  restricts to errors, and `ci.log_to('curie.log')` copies the session to a
  file. Fit summaries account for every candidate: nothing is dropped
  silently.
- **Configuration validation**: unknown `fit_config` keys warn with a
  closest-match suggestion (`unknown key 'SNR_Min' ignored - did you mean
  'SNR_min'?`); wrong types and out-of-range values raise precise errors.
  Opaque failure sites (empty counts, unparseable dates) now raise
  actionable messages.

### Added — diagnostics
- **`.diagnostics` DataFrames on Spectrum, Calibration and DecayChain**: one
  row per fit with reduced chi2, dof, points used/dropped, model tag,
  uncertainty scale factor, `flags` from a fixed vocabulary (`at_bound:<param>`,
  `chi2_high`, `fit_failed`, `unmoved`, `singular_cov`) and the full message
  text. `sp.fits` is now public.
- **Calibration point tables** `cb.engcal_data`/`rescal_data`/`effcal_data`:
  every measured point with `used`, `reason` and `residual` columns.
  Rejected points are kept (in separate storage groups, with reasons) rather
  than discarded; raw readers of the classic storage still see used points
  only.

### Added — DecayChain fitting
- **`dc.fit_config`** with four new opt-in count filters: `max_chi2` (on the
  originating peak fit's chi2, which `get_counts()` now carries into
  `dc.counts`), `exclude_lines` (isotope or (isotope, energy) with
  closest-line matching and typo-safe warnings), `time_range` (per-isotope
  decay-time windows) and `unc_R_floor` (relative floor on the returned
  uncertainty). `fit_R(p0={...})` seeds the fit in production-rate units.

### Added — calibration models
- **Selectable models via `cb.fit_config`**: energy gains `'cubic'` (with a
  numeric channel inverse and a non-monotonicity warning), resolution gains
  `'sqrt_quad'` (the Genie/InterSpec-family form), efficiency gains
  a log-log polynomial (`'loglog'`, or `'loglog-2'` through `'loglog-8'` for
  an explicit order) alongside forced or automatic Vidmar selection.
  Named threshold parameters replace the hardcoded cuts (`engcal/rescal/effcal_max_error`,
  `outlier_sigma`). The fitted model tags and efficiency energy range are
  saved in the calibration .json (older files load unchanged), and the
  efficiency warns once per calibration when evaluated beyond its fitted
  range.
- **`calibrate(eff_points=...)`**: user-supplied efficiency points join the
  fit with independent uncertainties — for extending the calibrated energy
  range or merging geometries.

### Changed — plots show excluded data
- `dc.plot()` draws a one-sigma band from the fit covariance and shows
  fit-excluded or threshold-excluded counts as grey open markers (previously
  hidden); axis limits are set by the used data, so a wild excluded point
  cannot flatten the plot. `sp.plot()` crosshatches the counts of failed
  multiplets in red (announced by a warning). Calibration plots show
  rejected points as grey open markers.

### Changed — behavior
- **Fit keyword arguments persist**: `fit_R`/`fit_A0`/`calibrate` kwargs
  merge into the instance's `fit_config` (as `fit_peaks` always has).
  Positional calls like `fit_R(0.25)` now raise TypeError — use
  `max_error=0.25`.
- `fit_A0` announces a singular covariance and substitutes the same finite
  fallback `fit_R` uses, instead of silently returning non-finite
  uncertainties.

### Fixed
- **Unresolvable same-isotope doublets straddling a channel rounding
  boundary were fit as two degenerate peaks** with an arbitrary intensity
  split, instead of merging with combined intensity. The 152Eu
  443.96/444.01 keV doublet (0.27 channels apart) split 9:1 onto the weak
  line, producing a 9x-wrong efficiency point inside the shipped example's
  calibration; with the fix, that spectrum's efficiency curve moves by up
  to ~6% and its production-rate fit's chi2/dof drops from ~1000 to ~50.
  **This is the one change to zero-configuration numerical results.** The
  identical-gamma proximity checks now compare float channel distance.
- Identical-gamma pairs with `A_bound < 1` could reject their whole
  multiplet ("initial guess is outside of provided bounds"), and a dropped
  second member fed the fit the dropped peak's parameters instead of the
  survivor's.
- The efficiency fit reported the post-inflation chi2/dof (which converges
  to ~1 by construction) instead of the actual goodness of fit.
- `np.float64` values in `fit_config` (e.g. a computed `I_min`) crashed the
  scalar/pair dispatch in `fit_peaks` and the isotope emission filters.

## [0.1.0] - 2026-07-07

Packaging and data-distribution release: no numerical results change. Nuclear
data moves out of the package directory into a per-user cache with verified,
on-demand downloads, and the package itself becomes a standard PEP 621
project.

### Changed — distribution
- **Nuclear data is fetched on first use** into a per-user data directory
  (`~/.local/share/curie` on Linux, `~/Library/Application Support/curie` on
  macOS, `%LOCALAPPDATA%\curie` on Windows; override with the
  `CURIE_DATA_DIR` environment variable — a data directory rather than a
  cache, so OS cache cleaners never evict a machine prepared for offline
  work). Files found in the platform's site-wide data directory (e.g.
  `/usr/local/share/curie`, `C:\ProgramData\curie`) are used read-only in
  place, so shared machines can be provisioned once by an administrator.
  Every download is SHA256-verified against `curie/data_registry.json`, which
  records the checksums of the exact files published in the GitHub data
  release. The setup.py post-install download hook is gone;
  `pip install curie` is now a small, fast, pure-Python install.
- **The ENDF and TENDL libraries are sharded per target**: looking up one
  reaction fetches a small per-target file (~2 MB for ENDF, ~0.1 MB for
  TENDL) instead of the whole library (748 MB endf.db, 37-56 MB per TENDL
  variant). Shards assemble incrementally into the local database;
  `ci.download()` still fetches each complete library in one
  checksum-verified file for offline use. Data releases live in the
  dedicated [curie-data](https://github.com/jtmorrell/curie-data)
  repository (one release tag per sharded library, keeping every release
  well under GitHub's 1000-assets cap), which also hosts the data tooling;
  the curie repository's releases are software releases only.
- **Data from earlier curie versions is adopted automatically**: files found
  in the old in-package data directory are checksum-verified and
  hardlinked/copied into the cache, so existing installations re-download
  nothing on upgrade.
- **Importing curie touches no database**: the preset-compound list and
  `Element` isotopic abundances now load on first access instead of at import
  or construction, so `import curie` is instant on a fresh machine and pure
  stopping-power work needs only the 184 KB ziegler.db.
- `Element.abundances` and `Element.isotopes` are now read-only properties
  (loaded from decay.db on first access); assigning to them was never
  supported and now raises.

### Changed — packaging
- PEP 621 `pyproject.toml` with the hatchling backend replaces `setup.py`;
  dependencies (numpy, scipy, pandas, matplotlib, pooch, platformdirs) are
  declared, the version is single-sourced from `curie/__init__.py`, and
  Python 2 compatibility code is removed. Requires Python >= 3.9.
- Releases publish to PyPI from CI via trusted publishing on `v*` version
  tags (wheel + sdist); data-release tags (`data-v*`) can never trigger a
  publish.

## [0.0.37] - 2026-07-06

Correlated-uncertainty release: `fit_R`, `fit_A0`, and `calibrate()` now
propagate the correlation structure of their inputs, so fitted uncertainties
no longer average away systematic floors. Reviewer-raised: production-rate
uncertainties could previously undercut the nuclear-data (gamma-intensity)
uncertainties feeding them (Monte Carlo coverage 13-35% vs 68% nominal;
64-68% after this release).

### Changed — numerical results
- **`fit_R`/`fit_A0` are generalized-least-squares fits** under a block
  covariance: counting errors are independent; gamma-intensity errors are
  common within a line, with an isotope-common fraction `norm_frac`
  (default 1.0) representing the decay-scheme normalization; efficiency
  errors are correlated within one calibration, using the calibration's
  parameter covariance when available. Central values can shift within
  their uncertainties (re-weighting; benchmark: thallium foils shift within
  1.1 sigma of the published rates), and uncertainties grow to include the
  correlated floors (thallium: 3.8-6.8% before, 8.3-18.6% after). A
  one-sided chi-square scale factor (disable with `scale_factor=False`)
  additionally inflates the uncertainty when the count data are mutually
  inconsistent; the inflation applies to the independent error component
  only — inconsistency between points cannot indict the correlated modes —
  iterated until the whitened chi-square is consistent with 1 (the
  uncalibrated example workflow reports 6.8% where the floors alone give
  1%, honestly pricing its chi2/dof of ~1000).
- **`calibrate()` fits the efficiency curve against its block covariance**
  (intensity common per line; decay constant and reference activity common
  per source), with absolute sigmas and the one-sided scale factor. The
  fitted curve re-weights (shipped-spectrum benchmark: up to ~7%), and
  `unc_eff` no longer averages below the source-activity floor.
- Counts assembled by `get_counts` carry optional uncertainty-decomposition
  columns (`unc_stat`, `unc_line`/`line`, `unc_corr`/`cal`, `energy`), and
  the peaks table records the efficiency-calibration identity (`effcal`
  column), so the correlated fits work from spectra and saved peak files
  alike. Bare counts with only totals fit diagonally with a printed
  warning; an opt-in uniform-correlation mode (`corr=`, `corr_group=`) and
  a full-covariance override (`cov=`) are available.
- Covariance magnitudes are taken from fitted model values rather than the
  measured counts, avoiding the downward bias of measurement-weighted
  correlated fits (Peelle's pertinent puzzle).

## [0.0.36] - 2026-07-03

Numerics-focused release: where 0.0.35 fixed crashes, 0.0.36 corrects numbers.
Changed results are called out below with their magnitude and reason —
re-check analyses that depend on them. A before/after benchmark (the shipped
152Eu spectrum end-to-end, plus the affected reactions and chains) is
summarized per item.

### Changed — numerical results
- **Activity/decay uncertainties no longer double-count the Poisson term,
  and the fit covariance now respects the counting floor.** The peak-fit
  covariance (weighted least squares with sigma=sqrt(counts)) already carries
  the full counting statistics of the fitted area — verified analytically and
  by Monte Carlo — so the extra 1/N term in `unc_decays` was a duplicate.
  The covariance is now absolute (`absolute_sigma=True`) with a one-sided
  scale factor: inflated by chi2/dof when above 1 (model mismatch, as
  before), but no longer deflated below the sqrt(N) counting floor when a
  clean low-background peak fits with chi2/dof < 1. Benchmark: -0.3% to
  -9.8% on the 152Eu lines. Central values are unchanged.
- **Un-resolvable gamma lines are now actually merged.** The intensity
  combination in `Spectrum._gammas` silently discarded the merged intensity
  (pandas chained assignment) and left triplets partially merged. Decays and
  decay rates derived from merged multiplets change accordingly (benchmark:
  the 152Eu 719.3 keV pair's combined intensity rises 38%, its inferred
  decays drop 27.5%; the eu fit_R production rate moves -0.8%).
- **No cross-section extrapolation beyond the evaluated grid** (`Reaction.
  interpolate` returns 0 off-grid for all libraries; ENDF, IRDFF, and the
  IAEA monitors previously extrapolated linearly). Threshold reactions queried off-grid stop returning
  fabricated values (90Zr(n,2n) at 30 MeV: 903.7 mb -> 0), and flux averages
  over windows extending past a grid edge lose the fabricated contribution
  (natTi(p,x)46Sc averaged over 40-95 MeV vs its 80 MeV grid ceiling: -22%).
- **Exactly and nearly equal decay constants use the confluent Bateman
  solution.** Chain members with relatively indistinguishable decay constants
  (below 1E-9) are evaluated with the exact degenerate-eigenvalue terms
  (t*exp(-lt) for a pair), replacing the 1E-12 floor. Real database ties
  (e.g. 154Hf->154Lu, both stored as 2.0 s) now give the analytic in-growth
  instead of 0.
- **Decay-chain composition is unit-independent and extends to true
  stability.** The chain-expansion cutoff compared decay constants to 1E-12
  in the user's time units, so chains depended on `units=` (99Ru dropped for
  'ns'; the 238U series never passed 234U for 's'). Chains now include all
  unstable members in every unit; long-lived members appear with
  correspondingly small activities, and e.g. the 152Eu chain now continues
  past 152Gd (1.1E14 y) to 140Ce.
- **Bateman evaluation is rescaled per branch** (decay constants divided by
  their geometric mean, time scaled inversely — an exact invariance). This
  keeps the partial-fraction products inside float64 for any time units
  (previously up to 1E302 in 1/Gy on the deep 238U branches) and makes
  activities unit-invariant at machine precision.
- **`decays()` is correct for slow members at long times.** The
  small-decay-constant branch selected the linear-in-time limit on lambda
  alone, but the expansion parameter is lambda*t: interior members of the
  extended 238U series (234U, 230Th, 226Ra) at geological times returned
  sign-wrong, orders-of-magnitude-wrong decay counts. The exponential
  differences are now evaluated with expm1, exact for every lambda, with no
  threshold branch (verified against direct integration of the activity).

### Fixed
- `.Spe` saving no longer truncates live/real times to whole seconds.
- The IEC reader no longer loses a final-line count value ending in 0 or 4.
- An all-zero spectrum (valid input) no longer crashes at construction.
- `_chi2` returns inf instead of a negative value for over-parameterized
  fit windows.
- `map_channel` raises a clear ValueError for non-invertible energy
  calibrations (zero slope, negative discriminant) instead of returning
  garbage int32 channels.
- The chemical-formula parser aggregates repeated elements (CH3OH gave two
  H rows).
- `Element._eff_Z_ratio` no longer raises NameError for protons when called
  directly.
- Chain deduplication uses a tuple key (string-join collided for two-digit
  member indices); `fit_R`/`fit_A0` bounds no longer produce NaN for a zero
  initial guess.
- `Isotope('Fe')` (bare element symbol) raises the natural-element guidance
  instead of an opaque parse error; isomer masses reconstructed from the
  mass excess use the ground-state convention; counts assigned to a stable
  isotope raise a clear ValueError; `cb=None` yields a default Calibration.

## [0.0.35] - 2026-07-01

Crash-fix (hotfix) release. Two fixes change numerical results; they are called
out below. All other numbers are unchanged from 0.0.34.

### Fixed
- `Spectrum(..., cb=...)` crashed for `Calibration` objects (mis-parenthesized
  type check).
- `Spectrum.saveas('*.Chn')` crashed for any spectrum with a real sample
  description; description and calibration fields are now padded to the fixed
  widths the reader expects, and zero-length descriptions no longer corrupt
  the record layout.
- Reading a `.Chn` file with 8192 or more channels truncated the histogram to
  zero length under numpy 2 (16-bit integer overflow in the reader).
- `235U(n,g)` and `235U(n,2n)` from the TENDL library crashed on interpolation
  (a duplicated energy point in the stored grid); duplicate energies are now
  dropped for the spline-interpolated TENDL libraries.
- `DecayChain.activity()`/`decays()` raised `KeyError` for stable chain
  members; they now return 0.
- Exactly equal decay constants in a chain produced inf/nan activities
  (`sign(0)=0` defeated the degeneracy guard). The tied pair's terms now
  cancel, matching the guard's behavior for nearly-equal constants (an exact
  confluent-eigenvalue solution is planned as a result-changing fix).
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
- Broken or stale documentation examples (`Reaction.interpolate`/`integrate`,
  `DecayChain`, `Element.range`, `Compound.range`, `Isotope.dose_rate`, and
  the decay example script's removed `N_plot` keyword).

### Changed — numerical results
- **`Compound.range()`** now integrates from 1 keV instead of 1 MeV, matching
  `Element.range()`. The old value at 2 MeV was ~36% low, with the discrepancy
  growing rapidly below that (old values below ~1 MeV were essentially
  meaningless); at 60 MeV the change is ~0.1%. The old values were wrong.
- **`Library.retrieve`** (and therefore `Reaction`) now sorts energy grids on
  load and drops exact duplicate rows. Several `iaea_monitors` reactions
  (e.g. natFe(p,x)51Cr, 98Mo(n,g)99Mo) store points out of order, which
  corrupted `integrate`/`average`. Results for those reactions change; the
  old values were wrong.
- **IAEA reactions with both a direct and a cumulative excitation function**
  for the same target and product (124Xe(p,pn)/(p,x)123Xe and
  176Yb(d,n)/(d,x)177Lu) previously returned the union of the two curves;
  each channel now returns its own data.

### Known data defects (deferred to the data-library rebuild)
- Three `iaea_monitors` tables carry defective rows that survive sorting:
  15N(p,n)15O and 98Mo(n,g)99Mo contain two conflicting copies of parts of
  their evaluations, and natFe(p,x)51Cr has one spurious point
  (4230 mb at 157 MeV) that distorts interpolation between 157.0 and
  157.1 MeV. These are stored-data defects and will be corrected in the
  rebuilt data libraries.

### Changed — packaging
- All nuclear-data downloads (`ci.download()` and the install hook) now fetch
  from the immutable `data-v1` GitHub release (SHA256s in
  `data_manifest.toml`) instead of Dropbox.
- `setup.py` now declares `install_requires` (numpy, scipy, pandas,
  matplotlib) and `python_requires>=3.9`.

### Added
- Public test suite (`tests/`, run with `python -m pytest`) and a CI workflow
  covering Python 3.9-3.13.
- `data_manifest.toml`: size and SHA256 records for the shipped nuclear-data
  databases, matching the `data-v1` release assets.

## [0.0.34] - 2026-05-06

### Changed
- `Reaction.average` and `Reaction.integrate` now use bin-center midpoint
  quadrature instead of trapezoidal integration, removing a foil-thickness
  dependence in flux-averaged cross sections when the flux spectrum contains
  endpoint spikes (e.g. post-stop residence pile-up from over-stopping foils
  in `Stack`). Docstring example values shift by ~1%.

## Earlier releases

Releases up to 0.0.33 predate this changelog.
