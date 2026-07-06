.. _quickinstall:

============
Installation
============

Curie is available through the `Python Package index`_, which allows installation using Python's standard command line utility `pip`_.  Assuming Python and Pip are installed already, you can install Curie with the command::

	pip install --user curie


or::

	python -m pip install --user curie


If you'd like to install the most recent (unreleased) version of Curie, you can clone the Curie `github`_ repository and install it from the source tree::

	git clone https://github.com/jtmorrell/curie.git
	python -m pip install --user ./curie

.. _Python Package index: https://pypi.org/
.. _pip: https://pip.pypa.io/en/stable
.. _github: https://github.com/jtmorrell/curie


Nuclear data
------------

Curie downloads the nuclear data files it needs the first time they are used:
each database is fetched from the `curie data release`_ on first access,
verified against its published SHA256 checksum, and stored in a per-user cache
directory.  The large ENDF cross-section library is fetched in small
per-target pieces, so looking up one reaction downloads a couple of MB rather
than the full 750 MB library.  No action is needed to make this happen —
the first ``ci.Reaction(...)`` or ``ci.Isotope(...)`` just works.

To prepare a machine for offline use (or to pre-populate the cache in one
step), download everything explicitly::

	import curie as ci
	ci.download()

If your data files are stale or corrupted, re-download them with::

	ci.download(overwrite=True)

The cache location follows your platform's convention (e.g.
``~/.cache/curie`` on Linux).  To place the data somewhere else — a shared
network drive, an air-gapped machine's data directory — set the
``CURIE_DATA_DIR`` environment variable; curie will use that directory for
all of its data.  On an air-gapped machine, copy the cache directory from a
connected machine (or the individual ``.db`` files from the `curie data
release`_) into ``CURIE_DATA_DIR``.

Data installed by curie versions before 0.1.0 (inside the package's own
directory) is found and adopted into the cache automatically — nothing is
re-downloaded after an upgrade.

.. _curie data release: https://github.com/jtmorrell/curie/releases/tag/data-v1
