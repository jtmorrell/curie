"""Shared machinery for the per-class fit-diagnostics tables.

Each fitting class (Spectrum, Calibration, DecayChain) exposes a `.diagnostics`
DataFrame with the same core columns; per-class extra columns are appended
after the core set.  The table is rebuilt on every fit, and is an empty frame
carrying the full schema before any fit has run.  Accessing it never triggers
a fit and never alters fit results.

Flag vocabulary (fixed): at_bound:<param>, unmoved, chi2_high,
singular_cov, extrapolated, non_monotonic, empty, fit_failed.
"""

import numpy as np
import pandas as pd

# Core columns shared by every class; per-class extras follow these.
_CORE_COLUMNS = {'fit': object, 'chi2': float, 'dof': int, 'n_points': int,
				 'n_dropped': int, 'converged': bool, 'model': object,
				 'scale_factor': float, 'flags': object, 'message': object}


def _diagnostics_frame(rows=None, extras=None):
	"""Assemble a diagnostics DataFrame: core schema plus per-class extras.

	`rows` is a list of dicts keyed by column name (each row must carry every
	column); `extras` is a dict of {column: dtype} appended after the core
	columns.  With no rows, returns an empty frame carrying the full schema.
	"""
	cols = dict(_CORE_COLUMNS)
	if extras is not None:
		cols.update(extras)
	if rows:
		return pd.DataFrame(rows, columns=list(cols))
	return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


def _at_bound(fit, lo, hi, rtol=1E-6):
	"""Fitted parameters that ended on their bounds.

	Returns a list of (index, 'lower'|'upper').  The tolerance is relative to
	the bound span, so it works for parameters of any scale; a one-sided or
	degenerate span falls back to exact equality (an infinite bound can never
	be hit).
	"""
	fit = np.asarray(fit, dtype=np.float64)
	lo = np.asarray(lo, dtype=np.float64)
	hi = np.asarray(hi, dtype=np.float64)
	out = []
	for n in range(len(fit)):
		span = hi[n]-lo[n]
		if not np.isfinite(span) or span<=0.0:
			if np.isfinite(lo[n]) and fit[n]==lo[n]:
				out.append((n, 'lower'))
			elif np.isfinite(hi[n]) and fit[n]==hi[n]:
				out.append((n, 'upper'))
			continue
		tol = rtol*span
		if fit[n]-lo[n]<=tol:
			out.append((n, 'lower'))
		elif hi[n]-fit[n]<=tol:
			out.append((n, 'upper'))
	return out


def _unmoved(fit, p0):
	"""True when the optimizer returned the starting estimate unchanged."""
	fit = np.asarray(fit, dtype=np.float64)
	p0 = np.asarray(p0, dtype=np.float64)
	return bool(len(fit)) and bool(np.allclose(fit, p0, rtol=1E-9, atol=0.0))
