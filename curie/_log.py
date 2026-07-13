"""Package-wide logging and shared configuration validation.

Every message curie emits on its own initiative (fit summaries, dropped-data
announcements, warnings) goes through the ``curie`` logger hierarchy defined
here.  Output requested by the user (e.g. ``summarize()``) stays on plain
``print``.

Conventions:

- Messages render as ``[LEVEL] Class.method: message`` on the console; the
  ``Class.method`` prefix is part of the message text, the ``[LEVEL]`` tag
  comes from the formatter.  Classes whose instances are commonly looped over
  name the instance in the locator - ``Spectrum(<filename>).method`` and
  ``DecayChain(<parent isotope>).method`` - so interleaved output stays
  attributable.
- Levels: ERROR accompanies a raised exception; WARNING means likely
  result-changing (dropped fitted results, failed fits, extrapolation);
  INFO is routine selection and per-fit summary lines (default console
  level); DEBUG carries per-candidate drop detail.
- The console handler writes to the *current* ``sys.stdout``, looked up at
  each message, for exact ``print`` parity under redirection and capture.
"""

import os
import sys
import difflib
import logging
import numbers

import numpy as np


class _StdoutHandler(logging.Handler):
	"""Stream handler that resolves sys.stdout at emit time (print parity)."""

	def emit(self, record):
		try:
			sys.stdout.write(self.format(record)+'\n')
		except Exception:
			self.handleError(record)


_CONSOLE_FORMAT = logging.Formatter('[%(levelname)s] %(message)s')
_FILE_FORMAT = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

_root = logging.getLogger('curie')
_root.setLevel(logging.DEBUG)
_root.propagate = False

_console = _StdoutHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_CONSOLE_FORMAT)
_root.addHandler(_console)

_file_handlers = {}


def _get_logger(module):
	"""Logger for a curie submodule, e.g. _get_logger('spectrum')."""
	return logging.getLogger('curie.'+module)


def _parse_level(level):
	if isinstance(level, numbers.Integral):
		return int(level)
	levels = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
			'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
	try:
		return levels[str(level).upper()]
	except KeyError:
		raise ValueError("Unknown log level {0!r}. Options: {1}.".format(
			level, ', '.join("'{}'".format(l) for l in levels)))


def set_log_level(level):
	"""Set the console message level

	Controls which curie messages are printed to the console.  Messages of
	the given level and above are shown.  The default level is 'INFO', which
	shows per-fit summary lines and warnings; 'DEBUG' additionally shows each
	dropped peak or count with the threshold that removed it; 'WARNING' shows
	only likely result-changing messages; 'ERROR' silences everything except
	messages accompanying raised exceptions.

	This affects the console only; log files created with `ci.log_to()` keep
	their own level.

	Parameters
	----------
	level : str
		One of 'DEBUG', 'INFO', 'WARNING', 'ERROR' (case-insensitive), or a
		numeric level from the logging module.

	Examples
	--------
	>>> ci.set_log_level('DEBUG')
	>>> ci.set_log_level('INFO')

	"""

	_console.setLevel(_parse_level(level))


def quiet_warnings():
	"""Silence curie's console messages

	One-call opt-out of curie's default console output: raises the console
	level to 'ERROR', silencing both the routine INFO summary lines and
	WARNING messages.  Errors accompanying raised exceptions are still shown.
	To keep warnings but silence the routine lines, use
	``ci.set_log_level('WARNING')`` instead.  Log files created with
	`ci.log_to()` are unaffected.

	Examples
	--------
	>>> ci.quiet_warnings()
	>>> ci.set_log_level('INFO')  # restore the default

	"""

	_console.setLevel(logging.ERROR)


def log_to(filename, level='INFO', mode='overwrite'):
	"""Write curie's messages to a log file

	Adds a log file receiving curie's messages at the given level,
	independent of the console level (e.g. a 'DEBUG' file under a quiet
	console).  By default an existing file at the same path is overwritten,
	since analysis scripts are typically re-run many times; pass
	``mode='append'`` to accumulate runs instead.  Calling `log_to` again
	with the same path replaces the previous file handler rather than
	duplicating messages.

	Parameters
	----------
	filename : str
		Path of the log file.

	level : str, optional
		Minimum message level written to the file: 'DEBUG', 'INFO',
		'WARNING' or 'ERROR'. Default 'INFO'.

	mode : str, optional
		'overwrite' (default) or 'append'.  The open()-style spellings
		'w' and 'a' are also accepted.

	Examples
	--------
	>>> ci.log_to('analysis.log')
	>>> ci.log_to('debug.log', level='DEBUG', mode='append')

	"""

	modes = {'overwrite': 'w', 'append': 'a', 'w': 'w', 'a': 'a'}
	if mode not in modes:
		raise ValueError("log_to: mode must be 'overwrite' or 'append' (got {0!r}).".format(mode))

	path = os.path.abspath(filename)
	if path in _file_handlers:
		_root.removeHandler(_file_handlers[path])
		_file_handlers[path].close()

	handler = logging.FileHandler(path, mode=modes[mode])
	handler.setLevel(_parse_level(level))
	handler.setFormatter(_FILE_FORMAT)
	_root.addHandler(handler)
	_file_handlers[path] = handler


########################
# Configuration validation
########################

def _is_number(v):
	return isinstance(v, numbers.Real)

def _is_int(v):
	# integer-valued floats count as integers: config values are often computed
	# (e.g. multi_max=8.0) and are only ever used in numeric comparisons
	if isinstance(v, numbers.Integral):
		return True
	return isinstance(v, numbers.Real) and float(v).is_integer()

def _is_number_or_pair(v):
	if isinstance(v, numbers.Real):
		return True
	try:
		return len(v)==2 and all(isinstance(i, numbers.Real) for i in v)
	except TypeError:
		return False


class _Check(object):
	"""One config key's validation: type predicate + description (+ choices)."""

	def __init__(self, predicate, description, allow_none=False):
		self.predicate = predicate
		self.description = description
		self.allow_none = allow_none

	def ok(self, v):
		if v is None:
			return self.allow_none
		return self.predicate(v)


def _choice(options):
	opts = {str(o).lower() for o in options}
	return _Check(lambda v: isinstance(v, str) and v.lower() in opts,
				'one of {}'.format(', '.join("'{}'".format(o) for o in sorted(opts))))


NUMBER = _Check(_is_number, 'a number')
NUMBER_OR_NONE = _Check(_is_number, 'a number or None', allow_none=True)
NUMBER_OR_PAIR = _Check(_is_number_or_pair, 'a number or 2-tuple of numbers', allow_none=True)
INTEGER = _Check(_is_int, 'an integer')
# np.bool_ is neither bool nor Integral, but flags computed from arrays
# (e.g. xrays=mask.any()) are legitimate config values
BOOLEAN = _Check(lambda v: isinstance(v, (bool, np.bool_, numbers.Integral)), 'a bool')


def _validate_config(updates, spec, context, logger):
	"""Validate a dict of config updates against a per-key spec.

	Unknown keys are ignored with a WARNING (and a closest-match suggestion
	when one exists); a value of the wrong type raises ValueError.  Returns
	the dict of accepted updates.
	"""

	accepted = {}
	for key, value in updates.items():
		if key not in spec:
			match = difflib.get_close_matches(key, spec.keys(), n=1, cutoff=0.6)
			if match:
				logger.warning("{0}: unknown key '{1}' ignored - did you mean '{2}'?".format(context, key, match[0]))
			else:
				logger.warning("{0}: unknown key '{1}' ignored".format(context, key))
			continue
		check = spec[key]
		if not check.ok(value):
			raise ValueError("{0}: '{1}' must be {2} (got {3} {4!r})".format(
				context, key, check.description, type(value).__name__, value))
		accepted[key] = value
	return accepted
