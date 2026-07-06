"""Split a cross-section library into per-target shard files for the nuclear-data release.

The ENDF and TENDL libraries store one table per target (e.g. CD_115, all
outgoing channels or residual products as columns) plus the small
all_reactions search index. Downloading a whole library to look up one target
is wasteful, so the data release carries one small sqlite file per table;
curie fetches only the shards a session actually touches and reassembles them
into its local data-directory database.

Usage:
    python scripts/shard_library.py <library> [--source curie/data/<library>.db] [--out shards/<library>]

where <library> is one of: endf, tendl, tendl_n_rp, tendl_p_rp, tendl_d_rp.

Writes <out>/<library>_<TABLE>.db, one per table, and updates the library's
shard group in curie/data_registry.json (per-shard SHA256s; the group's
base_url — its own GitHub release tag, so each release stays well under the
1000-assets-per-release limit — is preserved, or seeded for a new group). The
whole-file hash in the registry is pinned to the same source file, so the
eager and shard-assembled representations can never silently diverge. Table
names are validated as [A-Za-z0-9_] so the table<->filename mapping is
trivially invertible.
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


def shard(library, source, out_dir):
	os.makedirs(out_dir, exist_ok=True)
	# stale shards from a previous run would be uploaded as assets the
	# registry never references
	for old in os.listdir(out_dir):
		if old.startswith(library + '_') and old.endswith('.db'):
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
		fnm = '{}_{}.db'.format(library, tb)
		path = os.path.join(out_dir, fnm)
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
	ap.add_argument('library', choices=['endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp', 'tendl_d_rp'])
	ap.add_argument('--source', default=None)
	ap.add_argument('--out', default=None)
	args = ap.parse_args()
	source = args.source or os.path.join(root, 'curie', 'data', args.library + '.db')
	out = args.out or os.path.join(root, 'shards', args.library)

	registry_path = os.path.join(root, 'curie', 'data_registry.json')
	if not os.path.exists(registry_path):
		raise SystemExit('{} not found: run from a curie checkout (the registry carries the base_url and whole-file hashes).'.format(registry_path))
	with open(registry_path) as f:
		registry = json.load(f)

	# the whole-file entry and the shards are two representations of the same
	# library; pin both to this source so they can never silently diverge
	source_sha = sha256(source)
	recorded = registry['files'].get(args.library + '.db')
	if recorded and recorded != source_sha:
		print('NOTE: whole-file {}.db hash updated ({}... -> {}...); publish the new {}.db alongside the shards.'.format(
			args.library, recorded[:12], source_sha[:12], args.library))
	registry['files'][args.library + '.db'] = source_sha

	entries = shard(args.library, source, out)
	group = registry.setdefault('shards', {}).setdefault(args.library, {})
	if 'base_url' not in group:
		group['base_url'] = 'https://github.com/jtmorrell/curie/releases/download/data-v1-{}/'.format(args.library.replace('_', '-'))
		print('NOTE: new shard group; base_url set to {} - create that release tag before shipping.'.format(group['base_url']))
	group['files'] = entries
	with open(registry_path, 'w') as f:
		json.dump(registry, f, indent=1, sort_keys=True)
	print('registry updated: {} ({} {} shards)'.format(registry_path, len(entries), args.library))


if __name__ == '__main__':
	main()
