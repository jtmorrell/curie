"""Nuclear-data acquisition and connection layer.

Curie's databases live in a per-user data directory (~/.local/share/curie on
Linux, ~/Library/Application Support/curie on macOS, %LOCALAPPDATA%\\curie on
Windows; override with the CURIE_DATA_DIR environment variable) and are
fetched on first use from the GitHub data release recorded in
data_registry.json, with SHA256 verification. The ENDF and TENDL libraries
are additionally published as per-target shards (each library's shards on
their own release tag), so looking up one reaction downloads a fraction of a
MB instead of a 37-748 MB file; shards are assembled into the local library
database as they arrive.

Databases found in the platform's site-wide data directory (e.g.
/usr/local/share/curie, C:\\ProgramData\\curie) are used read-only in place, so
an administrator of a shared machine can populate that directory once and
every account works without downloading anything. Data installed by earlier
curie versions (in the package's own data directory) is adopted into the user
directory on first use, so existing installations re-download nothing.
"""

import os
import json
import hashlib
import shutil
import sqlite3

from ._log import _get_logger

_log = _get_logger('data')

GLOB_CONNECTIONS_DICT = {}

_REGISTRY_CACHE = None

_FILES = ['decay.db', 'ziegler.db', 'endf.db', 'tendl.db', 'tendl_n_rp.db',
		  'tendl_p_rp.db', 'tendl_d_rp.db', 'tendl_a_rp.db', 'IRDFF.db',
		  'iaea_monitors.db']

# accepted spellings beyond the filename stem itself
_ALIASES = {
	'tendl_n_rp.db': ['tendl_nrp', 'tendl_n', 'nrp', 'rpn'],
	'tendl_p_rp.db': ['tendl_prp', 'tendl_p', 'prp', 'rpp'],
	'tendl_d_rp.db': ['tendl_drp', 'tendl_d', 'drp', 'rpd'],
	'tendl_a_rp.db': ['tendl_arp', 'tendl_a', 'arp', 'rpa'],
	'iaea_monitors.db': ['iaea', 'iaea-cpr', 'iaea-monitor', 'cpr', 'iaea_cpr',
						 'iaea_monitor', 'medical', 'iaea-medical', 'iaea_medical'],
}


def _registry():
	global _REGISTRY_CACHE
	if _REGISTRY_CACHE is None:
		with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_registry.json')) as f:
			_REGISTRY_CACHE = json.load(f)
	return _REGISTRY_CACHE


def _canonical(db):
	"""Registry filename for a managed database name, or None for user paths."""
	nm = db.lower().replace('.db', '')
	for fnm in _FILES:
		if nm == fnm.lower().replace('.db', ''):
			return fnm
	for fnm, aliases in _ALIASES.items():
		if nm in aliases:
			return fnm
	return None


def _generation():
	"""The data generation this registry fetches: its release-tag name."""
	return _registry()['base_url'].rstrip('/').rsplit('/', 1)[-1]


def _check_generation(fnm, con):
	"""Warn when a local database is not from the generation this curie
	release fetches. Runs once per database per session (at first
	connection). ziegler.db carries no stamp by design (carried over
	between generations); a stampless file otherwise means data fetched by
	an earlier curie release. A mixed shard-assembled file cannot occur
	silently: assembly only stamps a database it created empty, so an
	older file keeps warning until it is replaced.
	"""
	if fnm == 'ziegler.db':
		return
	try:
		row = con.execute('SELECT generation FROM _version').fetchone()
	except sqlite3.Error:
		row = None
	expected = _generation()
	if row is None:
		_log.warning('{0} predates the generation stamp: it was fetched by an earlier '
					 'curie release (this one uses data generation {1}). Run '
					 'ci.download({2!r}) to update it.'.format(fnm, expected, fnm[:-3]))
	elif row[0] != expected:
		_log.warning('{0} is from data generation {1}, but this curie release uses '
					 'generation {2}. Run ci.download({3!r}) to update it.'.format(
						fnm, row[0], expected, fnm[:-3]))


def _data_path(db=''):
	"""The writable nuclear-data directory (CURIE_DATA_DIR overrides)."""
	path = os.environ.get('CURIE_DATA_DIR')
	if not path:
		import platformdirs
		path = platformdirs.user_data_dir('curie', appauthor=False)
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)
	return os.path.join(path, db) if db else path


def _site_data_paths(db=''):
	"""Site-wide data directories, consulted read-only for shared installs."""
	import platformdirs
	dirs = platformdirs.site_data_dir('curie', appauthor=False, multipath=True)
	return [os.path.join(d, db) for d in dirs.split(os.pathsep)]


def _site_file(fnm):
	"""Path of a usable site-wide copy of the file, or None."""
	for p in _site_data_paths(fnm):
		if os.path.isfile(p) and os.path.getsize(p) > 0:
			return p
	return None


def _legacy_data_path(db=''):
	"""Data directory used by curie < 0.1.0 (inside the installed package)."""
	return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', db))


def _db_available(db):
	"""True if the database can be opened without a network fetch."""
	fnm = _canonical(db)
	if fnm is None:
		return False
	for path in [_data_path(fnm)] + _site_data_paths(fnm) + [_legacy_data_path(fnm)]:
		if os.path.isfile(path) and os.path.getsize(path) > 0:
			return True
	return False


def _sha256(path):
	h = hashlib.sha256()
	with open(path, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			h.update(chunk)
	return h.hexdigest()


def _retrieve(fnm):
	"""Download one data-release asset into the data directory, verifying its SHA256.

	Whole files live on the main data release (top-level base_url); each
	sharded library's shards live on their own release tag (the group's
	base_url), keeping every release well under GitHub's assets-per-release
	limit.
	"""
	import pooch
	reg = _registry()
	url, known = reg['base_url'] + fnm, reg['files'].get(fnm)
	if known is None:
		for group in reg.get('shards', {}).values():
			if fnm in group['files']:
				url, known = group['base_url'] + fnm, group['files'][fnm]
				break
	_log.info('Downloading {} from the curie data release...'.format(fnm))
	return pooch.retrieve(url=url, known_hash=known, fname=fnm, path=_data_path())


def _adopt_legacy(fnm):
	"""Bring a pre-0.1.0 data file into the data directory if its checksum still matches."""
	legacy = _legacy_data_path(fnm)
	if not (os.path.isfile(legacy) and os.path.getsize(legacy) > 0):
		return False
	known = _registry()['files'].get(fnm)
	if known is not None and _sha256(legacy) != known:
		_log.warning('Existing {} is from an older data generation (or corrupted); it will be replaced from the current data release.'.format(fnm))
		return False
	dest = _data_path(fnm)
	try:
		os.link(legacy, dest)
	except OSError:
		shutil.copy2(legacy, dest)
	_log.info('Adopted existing {} from {}'.format(fnm, os.path.dirname(legacy)))
	return True


def _ensure_file(fnm):
	"""Make a managed database file available; returns the path to use.

	Lookup order: the user data directory (writable), then any site-wide data
	directory (used read-only in place), then data from a pre-0.1.0 install
	(adopted into the user directory), then a download. Sharded libraries
	(endf, the tendl variants) are not fetched whole here: absent a local
	copy, an empty database is created and populated per-target by
	_ensure_table.
	"""
	path = _data_path(fnm)
	if os.path.isfile(path) and os.path.getsize(path) > 0:
		return path
	site = _site_file(fnm)
	if site is not None:
		return site
	if _adopt_legacy(fnm):
		return path
	if fnm[:-3] in _registry().get('shards', {}):
		return path  # assembled lazily, shard by shard
	_retrieve(fnm)
	return path


def _ensure_table(db, table):
	"""Make `table` queryable in a managed database, fetching its shard if needed.

	No-op for whole-file databases, for site-provided copies (used as-is,
	never assembled into), and for unknown table names (the caller's own SQL
	then reports the missing table). Assembly is a single transaction, so an
	interrupted insert leaves no half-built table behind, and the write lock
	serializes concurrent assemblers of the same table.
	"""
	fnm = _canonical(db)
	if fnm is None:
		return
	group = _registry().get('shards', {}).get(fnm[:-3])
	if group is None:
		return
	if _ensure_file(fnm) != _data_path(fnm):
		return  # served from a site-wide copy: use as-is
	con = _get_connection(db)
	if table != '_version':
		# a database assembled from scratch carries the generation stamp
		# from its first table; an older stampless file is deliberately
		# left unstamped, so the generation check keeps warning about it
		# rather than a mixed-generation file passing as current
		empty = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 0
		if empty and '{}__version.db'.format(fnm[:-3]) in group['files']:
			_ensure_table(db, '_version')
	if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
		return
	shard_fnm = '{}_{}.db'.format(fnm[:-3], table)
	if shard_fnm not in group['files']:
		return
	shard_path = _retrieve(shard_fnm)
	con.execute("ATTACH DATABASE ? AS shard", (shard_path,))
	try:
		con.execute("BEGIN IMMEDIATE")
		try:
			# re-check under the write lock: a concurrent process may have
			# assembled this table since the check above
			if not con.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
				schema = con.execute("SELECT sql FROM shard.sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
				con.execute(schema)
				con.execute('INSERT INTO main."{0}" SELECT * FROM shard."{0}"'.format(table))
			con.commit()
		except:
			con.rollback()
			raise
	finally:
		con.execute("DETACH DATABASE shard")
	try:
		os.remove(shard_path)
	except OSError:
		pass  # a concurrent assembler may have removed it already


def download(db='all', overwrite=False):
	"""Download nuclear data files as sqlite .db files.

	Data files are fetched automatically the first time they are needed, so
	calling this function is only necessary to prepare an offline machine, to
	pre-populate a fresh data directory, or to repair/update existing data.
	An installed file from an older data generation is replaced by the
	current release automatically; `overwrite=True` replaces files
	unconditionally.  Files are stored in a per-user data directory
	(printed by `ci.data._data_path()`), which can be overridden with the
	`CURIE_DATA_DIR` environment variable.  Every download is verified against
	the SHA256 recorded for the curie data release.

	Note that the large cross-section libraries (ENDF, the TENDL variants) are
	normally assembled on demand from small per-target files; downloading them
	here fetches each complete library in one file.  Files already provided by
	a site-wide data directory are reported and skipped unless
	`overwrite=True`, which fetches a fresh copy into the user data directory
	(shadowing the site copy).

	Parameters
	----------
	db : str, optional
		Name of database to download, default is 'all'.  Options include:
		'all', 'decay', 'ziegler', 'endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp',
		'tendl_d_rp', 'tendl_a_rp', 'IRDFF', 'iaea_monitors'.

	overwrite : bool, optional
		If overwrite is `True`, will save write over existing data.  Default is `False`.

	Examples
	--------
	The most common use case will be to (re)download all the data files

	>>> ci.download(overwrite=True)

	Some other use cases:
	To update only the 'endf' library

	>>> ci.download('endf', True)

	Or to download the 'decay' library for the first time

	>>> ci.download('decay')

	"""

	if db.lower() in ['all', '*']:
		d = list(_FILES)
	else:
		fnm = _canonical(db)
		if fnm is None:
			raise ValueError("download: {0!r} is not a curie database. Options: {1}, or 'all'.".format(db, ', '.join(sorted(f[:-3] for f in _FILES))))
		d = [fnm]

	for fnm in d:
		path = _data_path(fnm)
		installed = os.path.isfile(path) and os.path.getsize(path) > 0
		if not installed and not overwrite:
			site = _site_file(fnm)
			if site is not None:
				if _sha256(site) == _registry()['files'].get(fnm):
					_log.info('{} is provided by the site-wide data directory; nothing to download.'.format(fnm))
					continue
				# a stale site-wide copy cannot be replaced in place (it may
				# be read-only); the fetched user-directory copy takes
				# precedence over it
				_log.info('{} in the site-wide data directory does not match the current data release; fetching a copy into the user data directory (which takes precedence).'.format(fnm))
		if installed and not overwrite:
			if _sha256(path) == _registry()['files'].get(fnm):
				_log.info("{0} already installed. Run ci.download('{0}', overwrite=True) to overwrite these files.".format(fnm.replace('.db', '')))
				continue
			# a file that no longer matches the current data release — an
			# older generation, a repaired file within one, or a partial
			# shard assembly — is replaced without needing overwrite=True:
			# this is the repair the generation warning points at
			_log.info('{} does not match the current data release; fetching the current file.'.format(fnm))
		# the existing file (if any) is left in place: the fetch verifies and
		# replaces it atomically, so a failed download never destroys data
		_stale_con = GLOB_CONNECTIONS_DICT.pop(path, None)
		if _stale_con is not None:
			_stale_con.close()
		try:
			_retrieve(fnm)
		except Exception as e:
			_log.warning('download: {}'.format(e))


def _get_connection(db='decay'):

	def connector(dbnm):
		if os.path.exists(dbnm):
			try:
				if os.path.getsize(dbnm) > 0:
					return sqlite3.connect(dbnm, timeout=30.0)
				else:
					raise ValueError('{} exists but is of zero size.'.format(dbnm))
			except:
				_log.error('Error connecting to {}.'.format(dbnm))
				if _data_path() in dbnm:
					_log.error('Try using ci.download("all", overwrite=True) to update nuclear data files.')
				raise
		else:
			_log.warning('database {} does not exist, creating new file.'.format(dbnm))
			return sqlite3.connect(dbnm, timeout=30.0)

	global GLOB_CONNECTIONS_DICT

	fnm = _canonical(db)
	if fnm is not None:
		path = _ensure_file(fnm)
		if path not in GLOB_CONNECTIONS_DICT:
			# a sharded library in the user directory legitimately starts as a
			# new or still-empty file (e.g. a session that connected but never
			# queried), to be populated per-target by _ensure_table
			fresh = (fnm[:-3] in _registry().get('shards', {}) and path == _data_path(fnm)
					 and (not os.path.exists(path) or os.path.getsize(path) == 0))
			# a generous lock timeout lets concurrent processes assembling
			# shards into the same database wait for each other
			GLOB_CONNECTIONS_DICT[path] = sqlite3.connect(path, timeout=30.0) if fresh else connector(path)
			if not fresh:
				# a fresh sharded file has no generation yet; it is stamped
				# at first assembly instead
				_check_generation(fnm, GLOB_CONNECTIONS_DICT[path])
		return GLOB_CONNECTIONS_DICT[path]

	else:
		if db not in GLOB_CONNECTIONS_DICT:
			GLOB_CONNECTIONS_DICT[db] = connector(db)
		return GLOB_CONNECTIONS_DICT[db]


def _get_cursor(db='decay'):
	conn = _get_connection(db)
	return conn.cursor()
