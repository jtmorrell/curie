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
	src = sqlite3.connect(source)
	tables = [r[0] for r in src.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()]
	schemas = dict(src.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall())
	src.close()

	entries = {}
	for n, tb in enumerate(sorted(tables)):
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
		if (n + 1) % 50 == 0 or n + 1 == len(tables):
			print('{}/{} shards written'.format(n + 1, len(tables)))
	return entries


def main():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	ap.add_argument('--source', default=os.path.join(root, 'curie', 'data', 'endf.db'))
	ap.add_argument('--out', default=os.path.join(root, 'shards'))
	args = ap.parse_args()

	registry_path = os.path.join(root, 'curie', 'data_registry.json')
	registry = {}
	if os.path.exists(registry_path):
		with open(registry_path) as f:
			registry = json.load(f)

	entries = shard(args.source, args.out)
	registry.setdefault('shards', {})['endf'] = entries
	with open(registry_path, 'w') as f:
		json.dump(registry, f, indent=1, sort_keys=True)
	print('registry updated: {} ({} endf shards)'.format(registry_path, len(entries)))


if __name__ == '__main__':
	main()
