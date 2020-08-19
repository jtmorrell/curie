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

	for fl in ['wwd6b1gk2ge5tgt/decay.db','tkndjqs036piojm/endf.db','zkoi6t2jicc9yqs/tendl.db','x2vfjr7uv7ffex5/tendl_d_rp.db','n0jjc0dv61j9of9/tendl_n_rp.db',
				'ib2a5lrhiwkcro5/tendl_p_rp.db','kq07684wtp890v5/ziegler.db','lzn8zs6y8zu3v0s/iaea_monitors.db','34sgcvt8n57b0aw/IRDFF.db']:
		fnm = fl.split('/')[1]
		if not os.path.isfile(path(fnm)):
			try:
				import urllib2
			except:
				import urllib.request as urllib2
			try:
				print('Downloading {}'.format(fnm))
				with open(path(fnm),'wb') as f:
					f.write(urllib2.urlopen('https://www.dropbox.com/s/{}?dl=1'.format(fl)).read())
			except:
				print('ERROR: Unable to download {}. See https://jtmorrell.github.io/curie/build/html/quickinstall.html for more help.'.format(fnm))


class install(_install):
	def run(self):
		_install.run(self)
		self.execute(_post_install, (self.install_lib,), msg="Downloading nuclear data files...")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='curie',
	  version='0.0.8',
	  description='Curie is a python toolkit to aid in the analysis of experimental nuclear data.',
	  long_description=long_description,
	  long_description_content_type="text/markdown",
	  url='https://github.com/jtmorrell/curie',
	  author='Jonathan Morrell',
	  author_email='jmorrell@berkeley.edu',
	  license='MIT',
	  packages=find_packages(),
	  include_package_data=True,
	  cmdclass={'install': install})#, 
	  #install_requires=['numpy', 'matplotlib', 'scipy', 'pandas'])