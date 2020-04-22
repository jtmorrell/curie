import curie as ci

def decay_examples():
	### 225RA decay chain, units of days, 9.0/day production rate, for 0.5 days
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
	dc.plot(N_plot=5)


def isotope_examples():
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


def extended_examples():

	ip = ci.Isotope('115INm')
	ip = ci.Isotope('Cu-67')
	print(ip.dose_rate(units='mR/h'))
	ip = ci.Isotope('58NI')
	print(ip.abundance)

	ip = ci.Isotope('135CEm')
	print(ip.dc)
	ip = ci.Isotope('235U')
	print(ip.get_NFY(E=0.0253))

	ip = ci.Isotope('221AT')
	print(ip.decay_const())
	print(ip.decay_const('m', True))

	ip = ci.Isotope('226RA')
	print(ip.half_life())
	print(ip.optimum_units())
	print(ip.half_life(ip.optimum_units()))

	ip = ci.Isotope('Co-60')
	print(ip.gammas(I_lim=1.0))

	ip = ci.Isotope('64CU')
	print(ip.gammas())
	print(ip.gammas(xrays=True, dE_511=1.0))

	ip = ci.Isotope('Pt-193m')
	print(ip.electrons(I_lim=5.0, E_lim=(10.0, 130.0)))

	ip = ci.Isotope('35S')
	print(ip.beta_minus())

	ip = ci.Isotope('18F')
	print(ip.beta_plus())

	ip = ci.Isotope('210PO')
	print(ip.alphas(I_lim=1.0))

	ip = ci.Isotope('Co-60')
	print(ip.dose_rate(activity=3.7E10, units='R/hr'))



	dc = ci.DecayChain('Ra-225', R=[[1.0, 1.0], [0.5, 1.5], [2.0, 6]], units='d')
	print(dc.R_avg)
	print(dc.isotopes)
	dc.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']
	sp.saveas('test_spec.json')

	dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
	dc.get_counts([sp], EoB='01/01/2016 08:39:08')
	dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')

	print(dc.decays('152EU', t_start=1, t_stop=2))
	print(dc.decays('152EU', t_start=50, t_stop=50.1, units='y'))
	print(dc.activity('152EU', time=0))
	print(dc.activity('152EU', time=13.537, units='y'))
	print(dc.fit_A0())
	dc.plot()

	dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
	dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')
	dc.get_counts([sp], EoB='01/01/2016 08:39:08')
	dc.fit_R()
	print(dc.fit_R())
	print(dc.R_avg)
	print(dc.isotopes)
	dc.plot()

	dc = ci.DecayChain('99MO', A0=350E6, units='d')
	dc.plot()

# isotope_examples()
decay_examples()
# extended_examples()