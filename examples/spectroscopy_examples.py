import curie as ci
import numpy as np

def basic_examples():
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
	sp.saveas('test.csv', replace=True)
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

def extended_examples():
	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']
	sp.isotopes = ['152EU', '40K']
	sp.fit_peaks(gammas=[{'energy':1460.8, 'intensity':10.66, 'unc_intensity':0.55}])
	sp.fit_peaks(gammas=ci.Isotope('40K').gammas(istp_col=True))
	sp.summarize()
	sp.saveas('test_spec.csv')
	sp.saveas('test_spec.db')
	sp.saveas('test_spec.json')
	sp.plot()

	cb = ci.Calibration()
	cb.calibrate([sp], [{'isotope':'152EU', 'A0':3.7E4, 'ref_date':'01/01/2016 12:00:00'}])
	cb.plot()
	cb.saveas('calib.json')
	sp.saveas('test_spec.json')


	cb = ci.Calibration()
	print(cb.engcal)
	print(cb.eng(np.arange(10)))
	cb.engcal = [0.1, 0.2, 0.003]
	print(cb.eng(np.arange(10)))

	cb = ci.Calibration()
	print(cb.effcal)
	print(cb.unc_effcal)
	print(cb.eff(50*np.arange(1,10)))
	print(cb.unc_eff(50*np.arange(1,10)))

	cb = ci.Calibration()
	print(cb.rescal)
	print(cb.res(100*np.arange(1,10)))

	cb = ci.Calibration()
	print(cb.engcal)
	print(cb.map_channel(300))
	print(cb.eng(cb.map_channel(300)))

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']

	cb = ci.Calibration()
	cb.calibrate([sp], sources=[{'isotope':'152EU', 'A0':3.5E4, 'ref_date':'01/01/2009 12:00:00'}])
	cb.plot_engcal()
	cb.plot_rescal()
	cb.plot_effcal()
	cb.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	print(sp.attenuation_correction(['Fe', ci.Compound('H2O', density=1.0)], x=[0.1, 0.5])(100*np.arange(1,10)))
	print(sp.attenuation_correction(['La', ci.Compound('Kapton', density=12.0)], ad=[0.1, 0.5])(100*np.arange(1,10)))

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	print(sp.geometry_correction(distance=4, r_det=5, thickness=0.1, sample_size=2, shape='square'))
	print(sp.geometry_correction(distance=30, r_det=5, thickness=10, sample_size=1))
	print(sp.geometry_correction(distance=4, r_det=5, thickness=0.1, sample_size=(2,1.5), shape='rectangle'))

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	print(sp.cb.engcal)
	sp.cb.engcal = [0.3, 0.184]
	sp.isotopes = ['152EU']
	sp.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.cb.engcal = [0.3, 0.1835]
	sp.isotopes = ['152EU']
	sp.auto_calibrate()
	print(sp.cb.engcal)
	sp.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.cb.engcal = [0.3, 0.1]
	sp.isotopes = ['152EU']
	sp.auto_calibrate(peaks=[[664, 121.8]])
	print(sp.cb.engcal)
	sp.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.cb.engcal = [0.3, 0.1]
	sp.isotopes = ['152EU']
	sp.auto_calibrate(guess=[0.3, 0.1835])
	print(sp.cb.engcal)
	sp.plot()

	sp = ci.Spectrum('eu_calib_7cm.Spe')
	sp.isotopes = ['152EU']
	sp.plot()
	sp.plot(xcalib=False)
	sp.plot(style='poster')
	sp.summarize()
	sp.saveas('test_plot.png')
	sp.saveas('eu_calib.Chn')
	sp.saveas('peak_data.csv')
	print(sp.fit_peaks(SNR_min=5, dE_511=12))
	print(sp.fit_peaks(bg='quadratic'))


basic_examples()
extended_examples()