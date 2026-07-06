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
verified against its published SHA256 checksum, and stored in a per-user data
directory.  The large cross-section libraries (ENDF and the TENDL variants)
are fetched in small per-target pieces, so looking up one reaction downloads
well under a MB rather than a 40-748 MB library.  No action is needed to
make this happen — the first ``ci.Reaction(...)`` or ``ci.Isotope(...)``
just works.

To prepare a machine for offline use (or to pre-populate the data directory
in one step), download everything explicitly::

	import curie as ci
	ci.download()

If your data files are stale or corrupted, re-download them with::

	ci.download(overwrite=True)

The data directory follows your platform's convention: ``~/.local/share/curie``
on Linux, ``~/Library/Application Support/curie`` on macOS, and
``%LOCALAPPDATA%\curie`` on Windows.  To place the data somewhere else — a
shared network drive, an air-gapped machine's data directory — set the
``CURIE_DATA_DIR`` environment variable; curie will use that directory for
all of its data.  On an air-gapped machine, copy the data directory from a
connected machine (or the individual ``.db`` files from the `curie data
release`_) into ``CURIE_DATA_DIR``.

On a shared (multi-user) machine, an administrator can instead populate the
site-wide data directory once — ``/usr/local/share/curie`` on Linux,
``/Library/Application Support/curie`` on macOS, ``C:\ProgramData\curie`` on
Windows — and curie will use those files read-only in place from every
account, with nothing downloaded or duplicated per user.  Files in the
site-wide directory (like those in ``CURIE_DATA_DIR``) are trusted as
provided: keep them in sync with the data release matching the installed
curie version when upgrading.

Data installed by curie versions before 0.1.0 (inside the package's own
directory) is found and adopted into the cache automatically — nothing is
re-downloaded after an upgrade.

.. _curie data release: https://github.com/jtmorrell/curie-data/releases
