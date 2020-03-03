.. _quickinstall:

============
Installation
============

Curie is available through the `Python Package index`_, which allows installation using Python's standard command line utility `pip`_.  Assuming Python and Pip are installed already, you can install Curie with the command::

	pip install --user curie


or::

	python -m pip install --user curie


If you'd like to install the most recent (unreleased) version of Curie, you can clone the Curie `github`_, and install by either running ``python -m setup.py install --user`` or ``make install``.

.. _Python Package index: https://pypi.org/
.. _pip: https://pip.pypa.io/en/stable
.. _github: https://github.com/jtmorrell/curie


Troubleshooting
---------------

Curie should download about 1GB of nuclear data during setup, however if this download fails for some reason the databases can be downloaded by downloading the `setup`_ file and running it with the command::

	python setup.py install --user

.. _setup: https://github.com/jtmorrell/curie/blob/master/setup.py

If this fails, you can alternatively download curie as a .zip file from `dropbox`_ and unzip it into the appropriate site-packages directory.  If you're unsure where that is, the following `StackExchange`_ is a useful reference.

.. _dropbox: https://www.dropbox.com/s/iohu07ing4e1b9r/curie.zip?dl=1
.. _StackExchange: https://stackoverflow.com/questions/122327/how-do-i-find-the-location-of-my-python-site-packages-directory/12950101