from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sqlite3

GLOB_CONNECTIONS_DICT = {}

def _data_path(db=''):
	return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', db))

def download(db='all', overwrite=False):
	"""Download nuclear data files as sqlite .db files.

	This function can be used to either download or update the nuclear data
	files required by Curie.  If you wish to update, or if the data files are
	corrupted, run `ci.download(overwrite=True)` to overwrite the existing data.

	Parameters
	----------
	db : str, optional
		Name of database to download, default is 'all'.  Options include:
		'all', 'decay', 'ziegler', 'endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp',
		'tendl_d_rp', 'IRDFF', 'iaea_monitors'.

	overwrite : bool, optional
		If overwrite is `True`, will save write over existing data.  Default is `False`.

	Examples
	--------
	The most common use case will be to (re)download all the data files

	>>> ci.download(overwrite=True)

	Some other use cases:
	To update only the 'endf' library

	>>> ci.download('endf', True)

	Or to download the 'decay' library for the first time

	>>> ci.download('decay')

	"""

	db = db.lower()
	if db in ['all','*']:
		d = ['decay', 'ziegler', 'endf', 'tendl', 'tendl_n_rp', 'tendl_p_rp', 'tendl_d_rp', 'IRDFF', 'iaea_monitors']
	elif db in ['decay']:
		d = ['decay']
	elif db in ['ziegler']:
		d = ['ziegler']
	elif db in ['endf']:
		d = ['endf']
	elif db in ['tendl']:
		d = ['tendl']
	elif db in ['tendl_n_rp', 'tendl_nrp', 'tendl_n', 'nrp', 'rpn']:
		d = ['tendl_n_rp']
	elif db in ['tendl_p_rp', 'tendl_prp', 'tendl_p', 'prp', 'rpp']:
		d = ['tendl_p_rp']
	elif db in ['tendl_d_rp', 'tendl_drp', 'tendl_d', 'drp', 'rpd']:
		d = ['tendl_d_rp']
	elif db in ['irdff']:
		d = ['IRDFF']
	elif db in ['iaea', 'iaea-cpr', 'iaea-monitor', 'cpr', 'iaea_cpr', 'iaea_monitor', 'medical', 'iaea-medical', 'iaea_medical']:
		d = ['iaea_monitors']
	else:
		print('db={} not recognized.'.format(db))
		return

	addr = {'decay':'wwd6b1gk2ge5tgt', 'endf':'tkndjqs036piojm', 'tendl':'zkoi6t2jicc9yqs', 'tendl_d_rp':'x2vfjr7uv7ffex5', 'tendl_n_rp':'n0jjc0dv61j9of9',
				'tendl_p_rp':'ib2a5lrhiwkcro5', 'ziegler':'kq07684wtp890v5', 'iaea_monitors':'lzn8zs6y8zu3v0s', 'IRDFF':'34sgcvt8n57b0aw'}
	
	if not os.path.isdir(_data_path()):
		os.mkdir(_data_path())

	try:
		import urllib2
	except:
		import urllib.request as urllib2

	for i in d:
		fnm = i+'.db'
		if (not os.path.isfile(_data_path(fnm))) or overwrite:
			
			try:
				print('Downloading {}'.format(fnm))
				with open(_data_path(fnm),'wb') as f:
					f.write(urllib2.urlopen('https://www.dropbox.com/s/{0}/{1}?dl=1'.format(addr[i],fnm)).read())
			except Exception as e:
					print(e)
		else:
			print("{0}.db already installed. Run ci.download('{0}', overwrite=True) to overwrite these files.".format(i))



def _get_connection(db='decay'):
	
	def connector(dbnm):
		if os.path.exists(dbnm):
			try:
				if os.path.getsize(dbnm)>0:
					return sqlite3.connect(dbnm)
				else:
					raise ValueError('{} exists but is of zero size.'.format(dbnm))
			except:
				print('Error connecting to {}.'.format(dbnm))
				if _data_path() in dbnm:
					print('Try using ci.download("all", overwrite=True) to update nuclear data files.')
				raise
		else:
			print('WARNING: database {} does not exist, creating new file.'.format(dbnm))
			return sqlite3.connect(dbnm)


	db_nm = db.lower().replace('.db','')
	db_f = None

	if db_nm in ['decay']:
		db_f = 'decay.db'

	elif db_nm in ['ziegler']:
		db_f = 'ziegler.db'

	elif db_nm in ['endf']:
		db_f = 'endf.db'

	elif db_nm in ['tendl']:
		db_f = 'tendl.db'

	elif db_nm in ['tendl_n_rp','tendl_nrp','tendl_n','nrp','rpn']:
		db_f = 'tendl_n_rp.db'

	elif db_nm in ['tendl_p_rp','tendl_prp','tendl_p','prp','rpp']:
		db_f = 'tendl_p_rp.db'

	elif db_nm in ['tendl_d_rp','tendl_drp','tendl_d','drp','rpd']:
		db_f = 'tendl_d_rp.db'

	elif db_nm in ['irdff']:
		db_f = 'IRDFF.db'

	elif db_nm in ['iaea', 'iaea-cpr', 'iaea-monitor', 'cpr', 'iaea_cpr', 'iaea_monitor', 'medical', 'iaea-medical', 'iaea_medical']:
		db_f = 'iaea_monitors.db'


	global GLOB_CONNECTIONS_DICT

	if db_f is not None:
		if db_f not in GLOB_CONNECTIONS_DICT:
			GLOB_CONNECTIONS_DICT[db_f] = connector(_data_path(db_f))
		return GLOB_CONNECTIONS_DICT[db_f]

	else:
		if db not in GLOB_CONNECTIONS_DICT:
			GLOB_CONNECTIONS_DICT[db] = connector(db)
		return GLOB_CONNECTIONS_DICT[db]



def _get_cursor(db='decay'):	
	conn = _get_connection(db)
	return conn.cursor()
