.. _plotter:

========
Plotting
========

.. autofunction:: curie.colormap
.. autofunction:: curie.set_style


Class Plotting Methods
======================

The following classes have plotting methods callable with the `cls.plot()` function, where `cls` is an instance of the appropriate class:

* Calibration
* Spectrum
* DecayChain
* Reaction
* Stack

All `cls.plot()` methods have no required arguments, and while some have specific optional arguments that are documented in the respective classes,
all plotting methods share the same optional **keyword arguments**, denoted in the code by `**kwargs`.

f : matplotlib.pyplot figure
	The figure to draw on.  Supply `f` *and* `ax` together to draw on top of existing axes (e.g. to overlay several plots).

ax : matplotlib.pyplot axes
	The axes to draw on.  Supply `f` *and* `ax` together to draw on top of existing axes.

figsize : tuple
	Width and height of the figure in inches, passed to `matplotlib.pyplot.subplots`, e.g. `(8, 5)`.

scale : str
	Can specify the scale for the x and y axes with the following options:

	* `log` or `logy` : Set *only* y-scale to log
	* `logx` : Set *only* x-scale to log
	* `lin`, `liny` or `linear` : Set *only* y-scale to linear
	* `linx` : Set *only* x-scale to linear
	* `linlin` : x-scale *and* y-scale linear
	* `loglog` : x-scale *and* y-scale log
	* `linlog` : linear x-scale and log y-scale
	* `loglin` : log x-scale and linear y-scale

show : bool
	Whether or not to show the figure using the matplotlib GUI.

saveas : str
	Full filename for saving the figure.  Any matplotlib-supported image format is accepted; a ``.pickle`` filename instead serializes the figure object itself, so it can be reloaded and edited later.

return_plot : bool
	If `True`, then a tuple of (fig, axes) will be returned when calling the `cls.plot()` method.  This can be used to draw multiple plots over each other, e.g. for plotting multiple cross-sections.

**style**, **palette** and **shade** are also available as keyword arguments, and have the same behavior as in `curie.set_style()`.
