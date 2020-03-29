from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

LIGHT = """#1abc9c #2ecc71 #3498db #9b59b6 #34495e #f1c40f #e67e22 #e74c3c #ecf0f1 #95a5a6
#81ecec #55efc4 #74b9ff #a29bfe #636e72 #ffeaa7 #fab1a0 #ff7675 #dfe6e9 #b2bec3
#7ed6df #badc58 #686de0 #e056fd #30336b #f6e58d #ffbe76 #ff7979 #dff9fb #95afc0
#00a8ff #4cd137 #273c75 #9c88ff #353b48 #fbc531 #e67e22 #e84118 #f5f6fa #7f8fa6
#00d2d3 #1dd1a1 #54a0ff #5f27cd #576574 #feca57 #e67e22 #ff6b6b #ecf0f1 #8395a7
#70a1ff #7bed9f #70a1ff #5352ed #57606f #eccc68 #ff7f50 #ff6b81 #ffffff #dfe4ea
#12CBC4 #C4E538 #0652DD #FDA7DF #833471 #FFC312 #EE5A24 #ED4C67 #ecf0f1 #95a5a6
#38ada9 #b8e994 #6a89cc #6a89cc #1e3799 #fad390 #fa983a #f8c291 #ecf0f1 #95a5a6
#2bcbba #26de81 #4b7bec #a55eea #778ca3 #fed330 #fd9644 #fc5c65 #ecf0f1 #d1d8e0
#9AECDB #55E6C1 #25CCF7 #D6A2E8 #3B3B98 #F8EFBA #FEA47F #FD7272 #ecf0f1 #CAD3C8
#63cdda #2ecc71 #778beb #cf6a87 #596275 #f7d794 #f3a683 #e77f67 #ecf0f1 #95a5a6
#34ace0 #33d9b2 #706fd3 #40407a #34495e #ffb142 #ff793f #ff5252 #f7f1e3 #d1ccc0
#34e7e4 #0be881 #4bcffa #575fcf #485460 #ffdd59 #ffc048 #ff5e57 #ecf0f1 #d2dae2
#7efff5 #32ff7e #18dcff #7d5fff #4b4b4b #fffa65 #ffaf40 #ff4d4d #ecf0f1 #95a5a6"""

DARK = """#16a085 #27ae60 #2980b9 #8e44ad #2c3e50 #f39c12 #d35400 #c0392b #bdc3c7 #7f8c8d
#00cec9 #00b894 #0984e3 #6c5ce7 #2d3436 #fdcb6e #e17055 #d63031 #b2bec3 #636e72
#22a6b3 #6ab04c #4834d4 #be2edd #130f40 #f9ca24 #f0932b #eb4d4b #c7ecee #535c68
#0097e6 #44bd32 #192a56 #8c7ae6 #2f3640 #e1b12c #d35400 #c23616 #dcdde1 #718093
#01a3a4 #10ac84 #2e86de #341f97 #222f3e #ff9f43 #d35400 #ee5253 #bdc3c7 #8395a7
#1e90ff #2ed573 #1e90ff #3742fa #2f3542 #ffa502 #ff6348 #ff4757 #f1f2f6 #ced6e0
#1289A7 #A3CB38 #1B1464 #D980FA #6F1E51 #F79F1F #EA2027 #B53471 #bdc3c7 #7f8c8d
#079992 #78e08f #4a69bd #4a69bd #0c2461 #f6b93b #e58e26 #e55039 #bdc3c7 #7f8c8d
#0fb9b1 #20bf6b #3867d6 #8854d0 #4b6584 #f7b731 #fa8231 #eb3b5a #bdc3c7 #a5b1c2
#BDC581 #58B19F #1B9CFC #82589F #182C61 #EAB543 #F97F51 #FC427B #bdc3c7 #2C3A47
#3dc1d3 #27ae60 #546de5 #c44569 #303952 #f5cd79 #f19066 #e15f41 #bdc3c7 #7f8c8d
#227093 #218c74 #474787 #2c2c54 #2c3e50 #cc8e35 #cd6133 #b33939 #aaa69d #84817a
#00d8d6 #05c46b #0fbcf9 #3c40c6 #1e272e #ffd32a #ffa801 #ff3f34 #bdc3c7 #808e9b
#67e6dc #3ae374 #17c0eb #7158e2 #3d3d3d #fff200 #ff9f1a #ff3838 #bdc3c7 #7f8c8d"""

def colormap(palette='default', shade='dark', aslist=False):
	"""Broad spectrum of hex-formatted colors for plotting

	Function to provide a broad palette of colors for plotting, inspired by the
	FlatUI color picker at https://flatuicolors.com/

	Parameters
	----------
	palette : str, optional
		Choice of color palette. Options are `default`, `american`, `aussie`,
		`british`, `canadian`, `chinese`, `french`, `german`, `indian`, `russian`,
		`spanish`, `swedish` and `turkish`.

	shade : str, optional
		`light` or `dark` theme.  Default is `dark`.

	aslist : bool, optional
		Specifies whether to return the colormap as a dictionary,
		e.g. {'k':'#000000', 'w':'#ffffff'} or as a list, e.g. ['#000000', '#ffffff'].
		Default is `False` (returns dictionary)

	Returns
	-------
	colormap : dict or list
		Colormap in list or dictionary format

	Examples
	--------
	>>> cm = ci.colormap()
	>>> print(cm)
	{'b': '#2980b9', 'g': '#27ae60', 'k': '#2c3e50', 'o': '#d35400', 'aq': '#16a085', 
	'p': '#8e44ad', 'r': '#c0392b', 'gy': '#7f8c8d', 'w': '#bdc3c7', 'y': '#f39c12'}
	>>> cm_list = ci.colormap(aslist=True)
	>>> cm_british_light = ci.colormap('british', 'light')

	"""

	pmap = ['default','american','aussie','british','canadian',
			'chinese','dutch','french','german','indian','russian',
			'spanish','swedish','turkish']
	cmap = ['aq','g','b','p','k','y','o','r','w','gy']

	if palette.lower() not in pmap:
		palette = 'default'

	if shade.lower()=='light':
		cl = LIGHT.split('\n')[pmap.index(palette.lower())].split(' ')
	else:
		cl = DARK.split('\n')[pmap.index(palette.lower())].split(' ')

	if aslist:
		return [cl[n] for n in [4,9,7,2,1,6,0,5,3]]

	return {cmap[n]:cl[n] for n in range(len(cl))}

def set_style(style='default', palette='default', shade='dark'):
	"""Preset styles for generating plots

	Adjusts default font sizes, colors and spacing for plots, with presets based 
	on the intended application (i.e. `poster`, `paper`, etc.)

	Parameters
	----------
	style : str, optional
		Preset style based on intended application of the plot. Options are 
		`paper`, `poster`, `presentation`, and the default: `default`.

	palette : str, optional
		Sets colormap palette for plots, see `colormap` for detailed options.

	shade : str, optional
		Sets colormap shade for plots, see `colormap` for detailed options

	Examples
	--------
	>>> ci.set_style('paper')
	>>> ci.set_style('poster', shade='dark')
	>>> ci.set_style('presentation', palette='aussie')

	"""

	from cycler import cycler

	cm = colormap(palette=palette, shade=shade)
	plt.rcParams['axes.prop_cycle'] = cycler(color=[cm['k'],cm['r'],cm['b'],cm['gy'],cm['g'],cm['p'],cm['o'],cm['aq'],cm['y']])

	plt.rcParams['font.family']='sans-serif'
	plt.rcParams['figure.autolayout'] = 'True'
	plt.rcParams['xtick.minor.visible']='True'
	plt.rcParams['ytick.minor.visible']='True'

	plt.rcParams['xtick.major.size']='5'
	plt.rcParams['xtick.minor.size']='3.5'
	plt.rcParams['ytick.major.size']='5'
	plt.rcParams['ytick.minor.size']='3.5'
	
	if style.lower()=='paper':
		plt.rcParams['font.size']='14'
		plt.rcParams['axes.labelsize']='16'
		plt.rcParams['legend.fontsize']='12'

		plt.rcParams['axes.titlepad']='10'
		plt.rcParams['xtick.major.pad']='6'
		plt.rcParams['ytick.major.pad']='6'
		
		plt.rcParams['xtick.major.width']='1.25'
		plt.rcParams['xtick.minor.width']='1.05'
		plt.rcParams['ytick.major.width']='1.25'
		plt.rcParams['ytick.minor.width']='1.05'

		plt.rcParams['axes.linewidth']='1.05'
		
		plt.rcParams['lines.markersize']='4.0'
		plt.rcParams['lines.linewidth'] = '2.2'

		plt.rcParams['errorbar.capsize']='4.5'
		plt.rcParams['lines.markeredgewidth']='1.2'

	elif style.lower()=='poster':
		plt.rcParams['font.size']='16'
		plt.rcParams['axes.labelsize']='18'
		plt.rcParams['legend.fontsize']='14'

		plt.rcParams['axes.titlepad']='12'
		plt.rcParams['xtick.major.pad']='8'
		plt.rcParams['ytick.major.pad']='8'

		plt.rcParams['xtick.major.width']='1.5'
		plt.rcParams['xtick.minor.width']='1.2'
		plt.rcParams['ytick.major.width']='1.5'
		plt.rcParams['ytick.minor.width']='1.2'

		plt.rcParams['axes.linewidth']='1.4'
		
		plt.rcParams['lines.markersize']='4.5'
		plt.rcParams['lines.linewidth'] = '2.8'

		plt.rcParams['errorbar.capsize']='5.0'
		plt.rcParams['lines.markeredgewidth']='1.4'

	elif style.lower()=='presentation':
		plt.rcParams['font.size']='15'
		plt.rcParams['axes.labelsize']='17'
		plt.rcParams['legend.fontsize']='12'

		plt.rcParams['axes.titlepad']='10'
		plt.rcParams['xtick.major.pad']='7'
		plt.rcParams['ytick.major.pad']='7'

		plt.rcParams['xtick.major.width']='1.25'
		plt.rcParams['xtick.minor.width']='1.05'
		plt.rcParams['ytick.major.width']='1.25'
		plt.rcParams['ytick.minor.width']='1.05'

		plt.rcParams['axes.linewidth']='1.15'
		
		plt.rcParams['lines.markersize']='4.0'
		plt.rcParams['lines.linewidth'] = '2.4'

		plt.rcParams['errorbar.capsize']='5.0'
		plt.rcParams['lines.markeredgewidth']='1.4'

	else: ### default
		plt.rcParams['font.size']='14'
		plt.rcParams['axes.labelsize']='15'
		plt.rcParams['legend.fontsize']='11'

		plt.rcParams['axes.titlepad']='8'
		plt.rcParams['xtick.major.pad']='5'
		plt.rcParams['ytick.major.pad']='5'

		plt.rcParams['xtick.major.width']='1.25'
		plt.rcParams['xtick.minor.width']='1.05'
		plt.rcParams['ytick.major.width']='1.25'
		plt.rcParams['ytick.minor.width']='1.05'

		plt.rcParams['axes.linewidth']='1.05'
		
		plt.rcParams['lines.markersize']='3.5'
		plt.rcParams['lines.linewidth'] = '2.0'

		plt.rcParams['errorbar.capsize']='4.2'
		plt.rcParams['lines.markeredgewidth']='1.15'



def _init_plot(**kwargs):
	if any([i in kwargs for i in ['style', 'palette', 'shade']]):
		st = kwargs['style'] if 'style' in kwargs else 'default'
		pa = kwargs['palette'] if 'palette' in kwargs else 'default'
		sh = kwargs['shade'] if 'shade' in kwargs else 'dark'
		set_style(style=st, palette=pa, shade=sh)


	f, ax = None, None
	if 'f' in kwargs and 'ax' in kwargs:
		f, ax = kwargs['f'], kwargs['ax']
	
	N = kwargs['_N_plots'] if '_N_plots' in kwargs else 1

	if f is None or ax is None:
		if 'figsize' in kwargs:
			f, ax = plt.subplots(1, N, figsize=kwargs['figsize'])
		else:
			f, ax = plt.subplots(1, N)

	return f, ax


def _draw_plot(fig, axis, **kwargs):
	f, ax = fig, axis

	if '_default_log' in kwargs:
		if kwargs['_default_log']:
			ax.set_yscale('log')
		else:
			ax.set_yscale('linear')

	if 'scale' in kwargs:
		s = kwargs['scale'].lower()
		if s in ['log','logy']:
			ax.set_yscale('log')
		elif s in ['lin','liny','linear']:
			ax.set_yscale('linear')
		elif s=='logx':
			ax.set_xscale('log')
		elif s=='linx':
			ax.set_xscale('linear')
		elif s in ['linlog','loglog','loglin','linlin']:
			ax.set_xscale(s[:3].replace('lin','linear'))
			ax.set_yscale(s[3:].replace('lin','linear'))

	f.tight_layout()

	if 'saveas' in kwargs:
		if kwargs['saveas'] is not None:
			f.savefig(kwargs['saveas'])


	if 'return_plot' in kwargs:
		if kwargs['return_plot']:
			return f, ax

	if 'show' in kwargs:
		if kwargs['show']:
			plt.show()
	else:
		plt.show()

	

	plt.close()
