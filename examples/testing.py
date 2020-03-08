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

el = ci.Element('La')
print(el.S(60.0))
print(el.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)

el = ci.Element('Fe')
print(el.range(60.0))
el = ci.Element('U')
print(el.range(60.0))