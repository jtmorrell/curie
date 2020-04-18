.. _reactions:

=========
Reactions
=========

Welcome to the Curie user's guide!  This section is under construction.  See :ref:`getting_started` for more info.

Reaction examples::

	rx = ci.Reaction('Ra-226(n,2n)Ra-225')
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