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

lb = ci.Library('tendl_p')
print(lb.search(target='Sr-86', product='Y-86g'))
lb = ci.Library('endf')
print(lb.search(target='226RA', product='225RA'))
print(lb.retrieve(target='226RA', product='225RA')[-8:])

# stack = [{'cm':'H20','ad':800.0,'name':'watr'},{'cm':'RbCl','density':3.0,'r':0.03,'name':'salt'},{'cm':'Kapton','r':0.025},{'cm':'Brass','ad':350,'name':'shiny'}]
# st = ci.Stack(stack, compounds='example_compounds.json')
# print st.stack
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
# sp.saveas('test_spec.json')

# rx = ci.Reaction('Ra-226(n,2n)Ra-225')
# rx.plot()

# dc = ci.DecayChain('Ra-225', R=[[1.0, 1.0], [0.5, 1.5], [2.0, 6]], units='d')
# print dc.R_avg
# dc.plot()

# dc = ci.DecayChain('152EU', A0=3.7E3, units='h')
# dc = ci.DecayChain('152EU', R=[[3E5, 36.0]], units='d')
# dc.get_counts(['eu_calib_7cm.Spe'], EoB='01/01/2016 08:39:08', peak_data='test_spec.json')
# dc.get_counts([sp], EoB='01/01/2016 08:39:08')
# print dc.fit_R()
# print dc.R_avg
# print dc.fit_A0()
# dc.plot()