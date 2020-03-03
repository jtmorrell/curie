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

ip = ci.Isotope('Co-60')
print(ip.dose_rate(activity=3.7E10, units='R/hr'))