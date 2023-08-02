.. _stopping:

===========================
Stopping Power Calculations
===========================

Curie can be used to calculate stopping powers using the Anderson-Ziegler formalism, and to retrieve photon
mass-attenuation coefficients, for any element up to Z=92, or any compound of elements.  The element
and compound classes can be used to directly calculate/retrieve these quantities.

Examples::

	el = ci.Element('Hf')
	print(el.mass)
	print(el.density)
	print(el.isotopes)
	print(el.abundances)
	f,ax = el.plot_mass_coeff(return_plot=True)
	el.plot_mass_coeff_en(energy=10.0**(np.arange(0,4,0.01)), f=f, ax=ax)
	el.plot_mass_coeff_en()

	el = ci.Element('Fe')
	print(el.attenuation(511, x=0.3))
	print(el.attenuation(300, x=0.5, density=8))
	print(el.S(20.0, particle='Ca-40'))
	print(el.S(60.0))
	el.plot_S(particle='40CA')
	print(el.range(60.0))
	el.plot_range()
	el.plot_mass_coeff()
	el.plot_mass_coeff(style='poster')
	ci.set_style()

	el = ci.Element('La')
	print(el.S(60.0))
	print(el.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)

	el = ci.Element('Fe')
	print(el.range(60.0))
	el = ci.Element('U')
	print(el.range(60.0))

	el = ci.Element('Hg')
	print(el.mu(200))
	print(el.mu_en(200))

	cm = ci.Compound('Silicone')
	print(cm.weights)

	for c in ['H2C3.2RbHeCe4Pb','H2O','NHO2','H2.5O1.5','SrCO3']:
		cm = ci.Compound(c)
		print(cm.weights)

	print('Silicone' in ci.COMPOUND_LIST)
	cm = ci.Compound('Silicone')
	print(list(map(str, cm.elements)))
	cm = ci.Compound('H2O', density=1.0)
	print(cm.mu(200))
	print(cm.mu_en(200))
	print(cm.weights)


	cm = ci.Compound('SS_316') # preset compound for 316 Stainless
	print(cm.attenuation(511, x=0.3))
	print(cm.attenuation(300, x=1.0, density=5.0))

	cm = ci.Compound('Fe')
	print(cm.range(60.0))
	cm = ci.Compound('SS_316')
	print(cm.range(60.0))

	cm = ci.Compound('Brass', weights={'Zn':-33,'Cu':-66})
	print(cm.weights)
	cm.saveas('compounds.csv')

	cm = ci.Compound('Bronze', weights={'Cu':-80, 'Sn':-20}, density=8.9)
	f,ax = cm.plot_range(return_plot=True)
	cm.plot_range(particle='d', f=f, ax=ax)

	cm = ci.Compound('Bronze', weights='example_compounds.json')
	print(cm.weights)
	cm.saveas('compounds.csv')

	cm = ci.Compound('Bronze', weights='example_compounds.csv', density=8.9)
	cm.plot_mass_coeff()
	cm.plot_S()
	cm.plot_range()

	cm = ci.Compound('SrCO3', density=3.5)
	print(cm.S(60.0))
	print(cm.S(55.0, density=1E-3)) ### S in MeV/(mg/cm^2)


Additionally, Curie can be used to determine the flux profile of particles through a "stack" of
material, that can be composed of either elements or compounds.  The transport calculation is done
using a predictor-corrector Monte Carlo method.  For more details, see the Curie :ref:`api`.

Examples::

	stack = [{'cm':'H2O', 'ad':800.0, 'name':'water'},
			{'cm':'RbCl', 'density':3.0, 't':0.03, 'name':'salt'},
			{'cm':'Kapton', 't':0.025},
			{'cm':'Brass','ad':350, 'name':'metal'}]

	st = ci.Stack(stack, compounds={'Brass':{'Cu':-66, 'Zn':-33}}, E0=60.0)
	st.saveas('example_stack.csv')
	st.saveas('example_stack.json', filter_name=False)
	st.saveas('example_stack.db', save_fluxes=False)
	st.summarize()
	st.summarize(filter_name=False)
	st.plot()
	st.plot(filter_name='salt')

	st = ci.Stack(stack, compounds='example_compounds.json')
	print(st.stack)
	st.saveas('stack_calc.csv')
	print(st.fluxes)
	st.saveas('test.csv')
	st.saveas('test.db')
	st.summarize()
	st.plot()

	st = ci.Stack('test_stack.csv')
	print(st.stack)
	st.plot()