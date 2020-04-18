import curie as ci
import numpy as np

def basic_examples():
	el = ci.Element('Fe')
	print(el.S(20.0))
	print(el.S(20.0, particle='a'))
	el.plot_S()


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

def extended_examples():

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

basic_examples()
extended_examples()