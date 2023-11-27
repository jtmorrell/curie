.. _getting_started:

===============
Getting Started
===============

Installation
------------

If you haven't already installed Curie, visit the :ref:`quickinstall` Guide.


Spectroscopy
------------

Curie provides two classes for spectroscopic analysis, the `Spectrum` class and the `Calibration` class. The following examples assume Curie has been imported as::

	import curie as ci

The following example, using the spectrum located on the Curie `github`_, demonstrates how to perform peak fits and an efficiency calibration::

	### Load and plot a spectrum
	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.plot()

	### Fit Europium Spectrum
	sp.isotopes = ['152EU']
	sp.plot()

	### Perform an efficiency calibration
	cb = ci.Calibration()
	cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.7E4, 'ref_date':'01/01/2009 12:00:00'}])

	### Save calibration
	cb.saveas('eu_calib.json')

	### This calibration can be re-loaded
	cb = ci.Calibration('eu_calib.json')
	### And manually assigned to any spectrum
	sp.cb = cb
	sp.cb.plot()

	### Print out peaks
	sp.summarize()

	### Save peak information
	sp.saveas('test.csv')
	### Save as .Chn format
	sp.saveas('eu_calib_7cm.Chn')

	### Plot ADC channels instead of energy
	sp.plot(xcalib=False)

	### Pick out a few peaks for manual calibration
	cb_data = [[664.5, 121.8],
				[1338.5, 244.7],
				[1882.5, 344.3],
				[2428, 444],
				[7698, 1408]]

	sp.auto_calibrate(peaks=cb_data)


	# ### Custom peaks
	sp.fit_peaks(gammas=[{'energy':1460.82, 'intensity':0.1066, 'unc_intensity':0.0017, 'isotope':'40K'}])
	sp.summarize()
	sp.plot()

	# ### More options with fits
	sp.fit_config = {'xrays':True, 'E_min':20.0, 'bg':'quadratic'}
	### Save and show the plot
	sp.plot(saveas='europium.png')

.. _github: https://github.com/jtmorrell/curie/blob/master/examples/


Stopping Power Calculations
---------------------------

Curie uses the Anderson & Ziegler formalism for calculating charged-particle stopping powers.  These stopping powers can be calculated on an element or compound basis::

	el = ci.Element('Fe')
	print(el.S(20.0))
	print(el.S(20.0, particle='a'))
	el.plot_S()


The `Stack` class allows one to calculate particle flux through a stack of foils using these stopping powers::

	stack = stack=[{'compound':'Ni', 'name':'Ni01', 'thickness':0.025},  # Thickness only (mm)
				{'compound':'Kapton', 'thickness':0.05},				# No name - will not be tallied
				{'compound':'Ti', 'name':'Ti01', 'thickness':1.025},  # Very thick: should see straggle
				{'compound':'Inconel','ad':1.0,'name':'test'},
				{'compound':'SrCO3', 'name':'SrCO3', 'area':0.785, 'mass':4.8E-3}]

	st = ci.Stack(stack, E0=45, particle='d', compounds=[{'Inconel':{'Fe':33, 'Ni':55}}])
	st.summarize()
	st.plot()

	### Import stack design from .csv file
	st = ci.Stack('test_stack.csv', particle='a', E0=70, min_steps=20, accuracy=1E-4)
	st.plot()

The file `test_stack.csv` used in this example can be found on the `curie github`_.

.. _curie github: https://github.com/jtmorrell/curie/blob/master/examples/


Decay Chains
------------

Curie has the capability of calculating/fitting to any possible decay chain, using the Bateman equations.  The following example demonstrates this for the radium-225 decay chain::

	dc = ci.DecayChain('225RA', units='d', R={'225RA':[[9, 0.5],[2, 1.5],[5,4.5]]})
	dc.plot()

	### Measured counts: [start_time (d), stop_time (d), decays, unc_decays]
	### Times relative to t=0 i.e. EoB time
	dc.counts = {'225AC':[[5.0, 5.1, 6E5, 2E4],
						  [6.0, 6.1, 7E5, 3E4]],
				'221FR':[5.5, 5.6, 6E5, 2E4]}

	### Find the scaled production rate that gives us these counts
	dc.fit_R()
	### Only plot the 5 most active isotopes in the decay chain
	dc.plot(max_plot=5)


Nuclear Data Libraries
----------------------

Curie contains data from the ENSDF, ENDF, IRDFF, IAEA-charged-particle and TENDL nuclear data libraries.  Information about a specific isotope, for example its half-life, can be retreieved using the `Isotope` class::

	i = ci.Isotope('60CO')
	i = ci.Isotope('Co-60')  # equivalent
	### Get LaTeX formatted name
	print(i.TeX)
	### Get isotope mass in amu
	print(i.mass)
	### Get half life in optimum units
	print(i.half_life(i.optimum_units(), unc=True), i.optimum_units())
	### Print DataFrame of the decay gammas
	print(i.gammas())
	### Print dose rate of 80 mCi at 30 cm
	print(i.dose_rate(activity=80*3.7E7, distance=30.0))

Nuclear reaction data can be searched for using the `Library` class, and used with the `Reaction` class::

	### We will plot the same reaction from two different libraries
	### Passing f,ax to rx.plot allows multiple plots on the same figure

	rx = ci.Reaction('90ZR(n,2n)89ZR', 'irdff')
	f,ax = rx.plot(return_plot=True, label='library')
	rx = ci.Reaction('90ZR(n,2n)89ZR', 'endf')
	rx.plot(f=f,ax=ax, label='library')


	### Compare (n,2n) and (n,3n) for endf vs tendl
	f, ax = None, None
	for lb in ['endf','tendl']:
		rx = ci.Reaction('226RA(n,2n)225RA', lb)
		f, ax = rx.plot(f=f, ax=ax, return_plot=True, label='both', energy=np.arange(0,30,0.1))
		rx = ci.Reaction('226RA(n,3n)224RA', lb)
		f, ax = rx.plot(f=f, ax=ax, return_plot=True, label='both', energy=np.arange(0,40,0.1))

	plt.show()

	# ### Search the TENDL-2015 neutron library for reactions producing 225RA from 226RA
	lb = ci.Library('tendl_n')
	print(lb.search(target='226RA', product='225RAg'))