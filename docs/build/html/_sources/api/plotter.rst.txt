.. _plotter:

========
Plotting
========

.. autofunction:: curie.colormap
.. autofunction:: curie.set_style


Class Plotting Method
=====================

The following classes have plotting methods callable with the `cls.plot()` function, where `cls` is an instance of the appropriate class:

* Calibration
* Spectrum
* DecayChain
* Reaction
* Stack

All `cls.plot()` methods have no required arguments, and while some have specific arguments that are documented in the respective classes,
all plotting methods share the same optional keyword arguments, denoted in the code by `**kwargs`.

**f**
: `matplotlib.pyplot figure`  
	If figure *and* axes are supplied, then plot will be drawn *on top* of those axes.

**ax**
: `matplotlib.pyplot axes`
	If figure *and* axes are supplied, then plot will be drawn *on top* of those axes.

**scale**
: `str`
	Can specify the scale for the x and y axes with the following options:

* `log` or `logy` : Set *only* y-scale to log
* `logx` : Set *only* x-scale to log
* `lin`, `liny` or `linear` : Set *only* y-scale to linear
* `linx` : Set *only* x-scale to linear
* `linlin` : x-scale *and* y-scale linear
* `loglog` : x-scale *and* y-scale log
* `linlog` : linear x-scale and log y-scale
* `loglin` : log x-scale and linear y-scale

**show**
: `bool`
	Whether or not to show the figure using the matplotlib GUI.

**saveas**
: `str`
	Full filename for saving the figure in one of the matplotlib supported filetypes.

**return_plot**
: `bool`
	If `True`, then a tuple of (fig, axes) will be returned when calling the `cls.plot()` method.  This can be used to draw multiple plots over each other, e.g. for plotting multiple cross-sections.
