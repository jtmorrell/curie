"""Split endf.db into per-target shard files for the nuclear-data release.

The ENDF library stores one table per target (e.g. CD_115, all outgoing channels
as columns) plus the small all_reactions search index. Downloading the whole
748 MB file to look up one target is wasteful, so the data release carries one
small sqlite file per table; curie fetches only the shards a session actually
touches and reassembles them into its local cache database.

Usage:
    python scripts/shard_endf.py [--source curie/data/endf.db] [--out <dir>]

Writes <out>/endf_<TABLE>.db, one per table, and updates
curie/data_registry.json with each shard's SHA256 (whole-file entries for the
other databases are preserved). Table names are validated as [A-Za-z0-9_] so
the table<->filename mapping is trivially invertible.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3


def sha256(path):
	h = hashlib.sha256()
	with open(path, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			h.update(chunk)
	return h.hexdigest()


def shard(source, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	# stale shards from a previous run would be uploaded as assets the
	# registry never references
	for old in os.listdir(out_dir):
		if old.startswith('endf_') and old.endswith('.db'):
			os.remove(os.path.join(out_dir, old))
	src = sqlite3.connect(source)
	# SQLite-internal tables (sqlite_sequence etc.) and schemaless rows are
	# not shardable content
	schemas = {n: s for n, s in src.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
			   if s and not n.startswith('sqlite_')}
	src.close()

	entries = {}
	for n, tb in enumerate(sorted(schemas)):
		if not re.fullmatch(r'[A-Za-z0-9_]+', tb):
			raise ValueError('table name {!r} is not filename-safe'.format(tb))
		fnm = 'endf_{}.db'.format(tb)
		path = os.path.join(out_dir, fnm)
		if os.path.exists(path):
			os.remove(path)
		con = sqlite3.connect(path)
		con.execute("ATTACH DATABASE ? AS src", (source,))
		con.execute(schemas[tb])
		con.execute('INSERT INTO "{0}" SELECT * FROM src."{0}"'.format(tb))
		con.commit()
		con.execute("DETACH DATABASE src")
		con.execute("VACUUM")
		con.close()
		entries[fnm] = sha256(path)
		if (n + 1) % 50 == 0 or n + 1 == len(schemas):
			print('{}/{} shards written'.format(n + 1, len(schemas)))
	return entries


def main():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	ap.add_argument('--source', default=os.path.join(root, 'curie', 'data', 'endf.db'))
	ap.add_argument('--out', default=os.path.join(root, 'shards'))
	args = ap.parse_args()

	registry_path = os.path.join(root, 'curie', 'data_registry.json')
	if not os.path.exists(registry_path):
		raise SystemExit('{} not found: run from a curie checkout (the registry carries the base_url and whole-file hashes).'.format(registry_path))
	with open(registry_path) as f:
		registry = json.load(f)

	# the whole-file entry and the shards are two representations of the same
	# library; pin both to this source so they can never silently diverge
	source_sha = sha256(args.source)
	recorded = registry['files'].get('endf.db')
	if recorded and recorded != source_sha:
		print('NOTE: whole-file endf.db hash updated ({}... -> {}...); publish the new endf.db alongside the shards.'.format(recorded[:12], source_sha[:12]))
	registry['files']['endf.db'] = source_sha

	entries = shard(args.source, args.out)
	registry.setdefault('shards', {})['endf'] = entries
	with open(registry_path, 'w') as f:
		json.dump(registry, f, indent=1, sort_keys=True)
	print('registry updated: {} ({} endf shards)'.format(registry_path, len(entries)))


if __name__ == '__main__':
	main()
