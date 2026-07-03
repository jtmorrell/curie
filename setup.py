from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
try:
	from setuptools import setup, find_packages
	from setuptools.command.install import install as _install
except:
	from distutils.core import setup
	from distutils.command.install import install as _install



def _post_install(loc):
	path = lambda db: os.path.realpath(os.path.join(loc,'curie','data',db))

	if not os.path.isdir(path('')):
		os.mkdir(path(''))

	for fnm in ['decay.db','endf.db','tendl.db','tendl_d_rp.db','tendl_n_rp.db',
				'tendl_p_rp.db','ziegler.db','iaea_monitors.db','IRDFF.db']:
		if not os.path.isfile(path(fnm)):
			try:
				import urllib2
			except:
				import urllib.request as urllib2
			try:
				print('Downloading {}'.format(fnm))
				with open(path(fnm),'wb') as f:
					f.write(urllib2.urlopen('https://github.com/jtmorrell/curie/releases/download/data-v1/{}'.format(fnm)).read())
			except:
				print('ERROR: Unable to download {}. See https://jtmorrell.github.io/curie/build/html/quickinstall.html for more help.'.format(fnm))


class install(_install):
	def run(self):
		_install.run(self)
		self.execute(_post_install, (self.install_lib,), msg="Downloading nuclear data files...")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='curie',
	  version='0.0.37',
	  description='Curie is a python toolkit to aid in the analysis of experimental nuclear data.',
	  long_description=long_description,
	  long_description_content_type="text/markdown",
	  url='https://github.com/jtmorrell/curie',
	  author='Jonathan Morrell',
	  author_email='jmorrell@berkeley.edu',
	  license='MIT',
	  packages=find_packages(),
	  include_package_data=True,
	  # nuclear data is distributed via GitHub data releases (data-v1), never in
	  # wheels/sdists: a wheel built from a dev tree with curie/data/ populated would
	  # otherwise swallow ~1 GB of .db files (and exceed PyPI's size limit)
	  exclude_package_data={'curie': ['data/*.db'], '': ['*.db']},
	  cmdclass={'install': install},
	  python_requires='>=3.9',
	  install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'])