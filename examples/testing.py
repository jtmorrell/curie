import numpy as np
import curie as ci

# ip = ci.Isotope('115INm')
# ip = ci.Isotope('Cu-67')
# print ip.dose_rate(units='mR/h')
# ip = ci.Isotope('58NI')
# print(ip.abundance)

# ip = ci.Isotope('135CEm')
# print(ip.dc)
# # ip = ci.Isotope('235U')
# # print(ip.get_NFY(E=0.0253))

# ip = ci.Isotope('221AT')
# print(ip.decay_const())
# print(ip.decay_const('m', True))

# ip = ci.Isotope('226RA')
# print(ip.half_life())
# print(ip.optimum_units())
# print(ip.half_life(ip.optimum_units()))

# ip = ci.Isotope('Co-60')
# print(ip.gammas(I_lim=1.0))

# ip = ci.Isotope('64CU')
# print(ip.gammas())
# print(ip.gammas(xrays=True, dE_511=1.0))

# ip = ci.Isotope('Pt-193m')
# print(ip.electrons(I_lim=5.0, E_lim=(10.0, 130.0)))

# ip = ci.Isotope('35S')
# print(ip.beta_minus())

# ip = ci.Isotope('18F')
# print(ip.beta_plus())

# ip = ci.Isotope('210PO')
# print(ip.alphas(I_lim=1.0))

# ip = ci.Isotope('Co-60')
# print(ip.dose_rate(activity=3.7E10, units='R/hr'))

# el = ci.Element('Hf')
# print(el.mass)
# print(el.density)
# print(el.isotopes)
# print(el.abundances)
# f,ax = el.plot_mass_coeff(return_plot=True)
# el.plot_mass_coeff_en(energy=10.0**(np.arange(0,4,0.01)), f=f, ax=ax)
# el.plot_mass_coeff_en()

# el = ci.Element('Fe')
# # print(el.attenuation(511, x=0.3))
# # print(el.attenuation(300, x=0.5, density=8))
# # print el.S(20.0, particle='Ca-40')
# print(el.S(60.0))
# # el.plot_S(particle='40CA')
# print(el.range(60.0))
# el.plot_range()
# # el.plot_mass_coeff()
# # el.plot_mass_coeff(style='poster')

# el = ci.Element('La')
# print(el.S(60.0))
# print(el.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)

# el = ci.Element('Fe')
# print(el.range(60.0))
# el = ci.Element('U')
# print(el.range(60.0))

# el = ci.Element('Hg')
# print(el.mu(200))
# print(el.mu_en(200))

# cm = ci.Compound('Silicone')
# print cm.weights

# for c in ['H2C3.2RbHeCe4Pb','H2O','NHO2','H2.5O1.5','SrCO3']:
# 	cm = ci.Compound(c)
# 	print cm.weights

# print 'Silicone' in ci.COMPOUND_LIST
# cm = ci.Compound('Silicone')
# print(list(map(str, cm.elements)))
# cm = ci.Compound('H2O', density=1.0)
# print(cm.mu(200))
# print(cm.mu_en(200))
# print(cm.weights)


# cm = ci.Compound('SS_316') # preset compound for 316 Stainless
# print(cm.attenuation(511, x=0.3))
# print(cm.attenuation(300, x=1.0, density=5.0))

# cm = ci.Compound('Fe')
# print(cm.range(60.0))
# cm = ci.Compound('SS_316')
# print(cm.range(60.0))

# cm = ci.Compound('Brass', weights={'Zn':-33,'Cu':-66})
# print cm.weights
# cm.saveas('compounds.csv')

# cm = ci.Compound('Bronze', weights={'Cu':-80, 'Sn':-20}, density=8.9)
# f,ax = cm.plot_range(return_plot=True)
# cm.plot_range(particle='d', f=f, ax=ax)

# cm = ci.Compound('Bronze', weights='example_compounds.json')
# print cm.weights
# cm.saveas('compounds.csv')

# cm = ci.Compound('Bronze', weights='example_compounds.csv', density=8.9)
# # cm.plot_mass_coeff()
# cm.plot_S()
# cm.plot_range()

# cm = ci.Compound('SrCO3', density=3.5)
# print(cm.S(60.0))
# print(cm.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)


# lb = ci.Library('tendl_n')
# print(lb.name)
# lb = ci.Library('endf')
# print(lb.name)

# lb = ci.Library('tendl_p')
# print(lb.search(target='Sr-86', product='Y-86g'))
# lb = ci.Library('endf')
# print(lb.search(target='226RA', product='225RA'))
# print(lb.retrieve(target='226RA', product='225RA')[-8:])

# rx = ci.Reaction('226RA(n,2n)')
# print(rx.library.name)
# rx = ci.Reaction('226RA(n,x)225RA')
# print(rx.library.name)
# rx = ci.Reaction('115IN(n,inl)')
# print(rx.library.name)

# stack = [{'cm':'H2O', 'ad':800.0, 'name':'water'},
# 		{'cm':'RbCl', 'density':3.0, 't':0.03, 'name':'salt'},
# 		{'cm':'Kapton', 't':0.025},
# 		{'cm':'Brass','ad':350, 'name':'metal'}]

# st = ci.Stack(stack, compounds={'Brass':{'Cu':-66, 'Zn':-33}}, E0=60.0)
# st.saveas('example_stack.csv')
# st.saveas('example_stack.json', filter_name=False)
# st.saveas('example_stack.db', save_fluxes=False)
# st.summarize()
# st.summarize(filter_name=False)
# st.plot()
# st.plot(filter_name='salt')

# st = ci.Stack(stack, compounds='example_compounds.json')
# print st.stack
# st.saveas('stack_calc.csv')
# print st.fluxes
# st.saveas('test.csv')
# st.saveas('test.db')
# st.summarize()
# st.plot()

# st = ci.Stack('test_stack.csv')
# print(st.stack)
# st.plot()


# sp = ci.Spectrum('eu_calib_7cm.Spe')
# sp.isotopes = ['152EU']
# sp.isotopes = ['152EU', '40K']
# sp.fit_peaks(gammas=[{'energy':1460.8, 'intensity':10.66, 'unc_intensity':0.55}])
# sp.fit_peaks(gammas=ci.Isotope('40K').gammas(istp_col=True))
# sp.summarize()
# sp.saveas('test_spec.csv')
# sp.saveas('test_spec.db')
# sp.saveas('test_spec.json')
# sp.plot()

# cb = ci.Calibration()
# cb.calibrate([sp], [{'isotope':'152EU', 'A0':3.7E4, 'ref_date':'01/01/2016 12:00:00'}])
# cb.plot()
# cb.saveas('calib.json')
# sp.saveas('test_spec.json')

# rx = ci.Reaction('Ra-226(n,2n)Ra-225')
# rx.plot()

# rx = ci.Reaction('Ni-58(n,p)')
# eng = np.linspace(1, 5, 20)
# phi = np.ones(20)
# print(rx.average(eng, phi))
# print(rx.average(eng, phi, unc=True))

# rx = ci.Reaction('115IN(n,g)')
# rx.plot(scale='loglog')
# rx = ci.Reaction('35CL(n,p)')
# f,ax = rx.plot(return_plot=True)
# rx = ci.Reaction('35CL(n,el)')
# rx.plot(f=f, ax=ax, scale='loglog')

# dc = ci.DecayChain('Ra-225', R=[[1.0, 1.0], [0.5, 1.5], [2.0, 6]], units='d')
# print dc.R_avg
# print dc.isotopes
# dc.plot()

# sp = ci.Spectrum('eu_calib_7cm.Spe')
# sp.isotopes = ['152EU']
# sp.saveas('test_spec.json')

# dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
# dc.get_counts([sp], EoB='01/01/2016 08:39:08')
# dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')

# print dc.decays('152EU', t_start=1, t_stop=2)
# print dc.decays('152EU', t_start=50, t_stop=50.1, units='y')
# print dc.activity('152EU', time=0)
# print dc.activity('152EU', time=13.537, units='y')
# dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
# dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')
# dc.get_counts([sp], EoB='01/01/2016 08:39:08')
# dc.fit_R()
# print dc.fit_R()
# print dc.R_avg
# print dc.isotopes
# print dc.fit_A0()
# dc.plot()

# dc = ci.DecayChain('99MO', A0=350E6, units='d')
# dc.plot()

# cb = ci.Calibration()
# print(cb.engcal)
# print(cb.eng(np.arange(10)))
# cb.engcal = [0.1, 0.2, 0.003]
# print(cb.eng(np.arange(10)))

# cb = ci.Calibration()
# print(cb.effcal)
# print(cb.unc_effcal)
# print(cb.eff(50*np.arange(1,10)))
# print(cb.unc_eff(50*np.arange(1,10)))

# cb = ci.Calibration()
# print(cb.rescal)
# print(cb.res(100*np.arange(1,10)))

# cb = ci.Calibration()
# print(cb.engcal)
# print(cb.map_channel(300))
# print(cb.eng(cb.map_channel(300)))

# sp = ci.Spectrum('eu_calib_7cm.Spe')
# sp.isotopes = ['152EU']

# cb = ci.Calibration()
# cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
# cb.plot_engcal()
# cb.plot_rescal()
# cb.plot_effcal()
# cb.plot()

sp = ci.Spectrum('eu_calib_7cm.Spe')
print(len(sp.hist))
sp.rebin(1000)
print(len(sp.hist))