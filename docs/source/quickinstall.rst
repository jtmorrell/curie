.. _quickinstall:

============
Installation
============

Curie is available through the `Python Package Index`_, which allows installation using Python's standard command line utility `pip`_.  Assuming Python and pip are installed already, you can install Curie with the command::

	pip install --user curie


or::

	python -m pip install --user curie


If you'd like to install the most recent (unreleased) version of Curie, you can clone the Curie `GitHub`_ repository and install it from the source tree::

	git clone https://github.com/jtmorrell/curie.git
	python -m pip install --user ./curie

Each release's built wheel is also attached to the corresponding `GitHub
release`_, for machines that cannot reach PyPI — see
:ref:`restricted_networks` below.

.. _Python Package Index: https://pypi.org/
.. _pip: https://pip.pypa.io/en/stable
.. _GitHub: https://github.com/jtmorrell/curie
.. _GitHub release: https://github.com/jtmorrell/curie/releases


Nuclear data
------------

Curie downloads the nuclear data files it needs the first time they are used:
each database is fetched from the `curie data release`_ on first access,
verified against its published SHA256 checksum, and stored in a per-user data
directory.  The large cross-section libraries (ENDF and the TENDL variants)
are fetched in small per-target pieces, so looking up one reaction downloads
well under a MB rather than a 37-748 MB library.  No action is needed to
make this happen — the first ``ci.Reaction(...)`` or ``ci.Isotope(...)``
just works.

To download everything up front (for offline use, or to pre-populate the data
directory in one step)::

	import curie as ci
	ci.download()

If your data files are stale or corrupted, re-download them with::

	ci.download(overwrite=True)

The data directory follows your platform's convention:

- **Windows**: ``%LOCALAPPDATA%\curie`` — almost always
  ``C:\Users\<username>\AppData\Local\curie``
- **Linux**: ``~/.local/share/curie``
- **macOS**: ``~/Library/Application Support/curie``

Data installed by Curie versions before 0.1.0 (inside the package's own
directory) is found and adopted into the data directory automatically —
nothing is re-downloaded after an upgrade.

.. _curie data release: https://github.com/jtmorrell/curie-data/releases


Troubleshooting
---------------

.. _restricted_networks:

Installing behind a restrictive proxy or firewall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On some networks a proxy blocks pip, conda, and Curie's automatic data
downloads, while ordinary browser downloads from github.com still work.
Curie can be installed entirely by hand in that situation — no environment
variables or administrator rights are needed.

**Installing the code.**  Download the wheel (the ``.whl`` file) from the
latest `GitHub release`_ in a browser and install it locally::

	pip install path/to/curie-X.Y.Z-py3-none-any.whl

Installing from a local wheel requires no network access.  If pip cannot
reach PyPI to resolve dependencies, add ``--no-deps`` — Curie's dependencies
(numpy, scipy, pandas, matplotlib, pooch, platformdirs) are all included in
a standard Anaconda installation.  Prefer the wheel over copying the source
tree into ``site-packages`` by hand: a hand-copied tree has no package
metadata, so pip cannot see, upgrade, or uninstall it.

**Installing the data.**  Run any Curie command that needs data once, e.g.::

	import curie as ci
	ci.download()

The download will fail, but this creates the data directory, and the error
message reports its location along with the data release page to fetch from.
Download the ``.db`` files from the `curie data release`_ page in a browser
and place them in that directory.  The downloads can be verified against the
``SHA256SUMS`` file published on the same release page.

On Windows the data directory is ``%LOCALAPPDATA%\curie`` — almost always
``C:\Users\<username>\AppData\Local\curie``.  The ``AppData`` folder is
hidden by default in File Explorer, but typing ``%LOCALAPPDATA%\curie``
directly into the Explorer address bar (or the Win+R run dialog) opens it.

**Alternatives.**  On any machine that can reach GitHub, ``ci.download()``
populates the data directory; copying that directory to the same location on
the restricted machine works just as well as downloading by hand.  To place
the data somewhere else — a shared network drive, a project directory — set
the ``CURIE_DATA_DIR`` environment variable to that folder before starting
Python (a per-session ``set CURIE_DATA_DIR=...`` in cmd, or
``$env:CURIE_DATA_DIR = '...'`` in PowerShell, requires no administrator
rights).  On a shared (multi-user) machine, an administrator can instead
populate the site-wide data directory once — ``/usr/local/share/curie`` on
Linux, ``/Library/Application Support/curie`` on macOS,
``C:\ProgramData\curie`` on Windows — and Curie uses those files read-only
in place from every account, with nothing downloaded or duplicated per user.
Files placed manually (in the data directory, ``CURIE_DATA_DIR``, or the
site-wide directory) are trusted as provided: keep them in sync with the
data release matching the installed Curie version when upgrading.
