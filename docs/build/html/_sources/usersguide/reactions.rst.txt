.. _reactions:

=========
Reactions
=========

Curie provides access to the following evaluated nuclear reaction libraries: ENDF/B-VII.1, TENDL-2015, IRDFF-II,
and the IAEA Medical Monitor reaction library.  The `Library` class gives access to the libraries for searching 
and retrieving reactions.  The `Reaction` class gives access to data and methods for a specific reaction.  Some 
methods include the flux-average cross section, the integral of the cross section and the flux, a plotting method,
and interpolation.  See the :ref:`api` for more details.

Examples::

	rx = ci.Reaction('Ra-226(n,2n)Ra-225', 'endf')
	rx.plot()

	rx = ci.Reaction('Ni-58(n,p)')
	eng = np.linspace(1, 5, 20)
	phi = np.ones(20)
	print(rx.average(eng, phi))
	print(rx.average(eng, phi, unc=True))

	rx = ci.Reaction('115IN(n,g)')
	rx.plot(scale='loglog')
	rx = ci.Reaction('35CL(n,p)')
	f,ax = rx.plot(return_plot=True)
	rx = ci.Reaction('35CL(n,el)')
	rx.plot(f=f, ax=ax, scale='loglog')