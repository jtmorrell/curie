"""Laziness guarantees for the data layer: importing curie touches no database,
and pure stopping-power work needs only ziegler.db.

These run curie in a subprocess with CURIE_DATA_DIR pointed at a fresh
directory: any database access shows up there as an adopted/fetched file (or
as a failed fetch on a machine with no local data), so an empty cache after
the run proves nothing was touched.
"""
import os
import shutil
import subprocess
import sys

import pytest

from conftest import requires_data


def _run(code, cache):
	env = dict(os.environ, CURIE_DATA_DIR=str(cache))
	return subprocess.run([sys.executable, '-c', code], env=env,
						  capture_output=True, text=True, timeout=120)


def test_import_touches_no_database(tmp_path):
	r = _run('import curie', tmp_path)
	assert r.returncode == 0, r.stderr
	assert os.listdir(str(tmp_path)) == []


@requires_data('ziegler')
def test_compound_list_lazy_and_populated(tmp_path):
	r = _run('import curie as ci\n'
			 'assert "Silicone" in ci.COMPOUND_LIST\n'
			 'assert "Kapton" in ci.COMPOUND_LIST', tmp_path)
	assert r.returncode == 0, r.stderr
	assert sorted(os.listdir(str(tmp_path))) == ['ziegler.db']


@requires_data('ziegler')
def test_element_stopping_power_needs_only_ziegler(tmp_path):
	code = ('import curie as ci\n'
			'el = ci.Element("Fe")\n'
			'assert abs(el.mass - 55.847) < 0.01\n'
			'assert el.S(20.0) > 0\n'
			'assert el.mu(100.0) > 0\n')
	r = _run(code, tmp_path)
	assert r.returncode == 0, r.stderr
	assert sorted(os.listdir(str(tmp_path))) == ['ziegler.db']


@requires_data('ziegler', 'decay')
def test_element_abundances_lazy(tmp_path):
	code = ('import curie as ci\n'
			'el = ci.Element("Fe")\n'
			'assert "56FE" in el.isotopes\n'
			'assert abs(el.abundances["abundance"].sum() - 100.0) < 0.1\n')
	r = _run(code, tmp_path)
	assert r.returncode == 0, r.stderr
	assert sorted(os.listdir(str(tmp_path))) == ['decay.db', 'ziegler.db']
