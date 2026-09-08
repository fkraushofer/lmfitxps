.. _PeakModels:

Peak-like/Step-like models
==========================
.. index:: Peak-Models, Fermi edge, Singlet, Doublet

The following sections document the peak-like/step-like models implemented as an extension to the `lmfit built-in models <https://lmfit.github.io/lmfit-py/builtin_models.html>`_.
The models are thereby mostly based on the `lmfit lineshapes module <https://github.com/lmfit/lmfit-py/blob/master/lmfit/lineshapes.py>`_.

Energy convention
-----------------

Supply binding energies in descending order or kinetic energies in ascending
order. The models infer the energy scale from this ordering; inverting a plot
axis does not change the model convention. For the same spectrum, converting
with ``binding_energy = energy_offset - kinetic_energy`` leaves the intensity
array unchanged and transforms each peak center by the same relation.

For the Doniach-Sunjic singlet and doublet, ``gamma`` is the asymmetry
parameter. Positive values produce a tail toward higher binding energy,
equivalently lower kinetic energy. The doublet's positive ``soc`` places the
secondary peak at higher binding energy or lower kinetic energy than the
primary peak. Widths, asymmetry, amplitudes, and splitting retain their values
when converting between the two energy scales.

.. note::

   Binding-energy Doniach profiles are reflected about each component's center.
   Versions up to and including 4.2.0 incorrectly placed their tails toward
   lower binding energy. Binding-energy fits made with these versions should
   be refitted; negating ``gamma``
   does not mirror a Doniach profile and can produce negative intensities.


.. _FermiEdgeModel:

:py:class:`FermiEdgeModel`
__________________________

.. autoclass:: lmfitxps.models.FermiEdgeModel
    :exclude-members: guess, __init__, _set_paramhints_prefix
    :noindex:

.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.

.. _ConvGaussianDoniachSinglett:
.. _ConvGaussianDoniachSinglet:

:py:class:`ConvGaussianDoniachSinglet`
_______________________________________

.. autoclass:: lmfitxps.models.ConvGaussianDoniachSinglet
    :exclude-members: guess, __init__, _set_paramhints_prefix
    :noindex:
.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.

.. _ConvGaussianDoniachDublett:
.. _ConvGaussianDoniachDoublet:

:py:class:`ConvGaussianDoniachDoublet`
______________________________________

.. autoclass:: lmfitxps.models.ConvGaussianDoniachDoublet
    :exclude-members: guess, __init__, _set_paramhints_prefix
    :noindex:

.. note::
   The class functions are inherited from the lmfit Model class. For details, please refer to their documentation at
   `lmfit Model Class Methods <https://lmfit.github.io/lmfit-py/model.html#model-class-methods>`_.

Compatibility aliases
---------------------

The historical names ``ConvGaussianDoniachSinglett`` and
``ConvGaussianDoniachDublett`` remain exact aliases for backwards
compatibility. The corresponding ``eval_dublett_components`` method and
``dublett_ratio_diagnostics`` and ``dublett_ratio_report`` helpers are also
retained. New code should use the correctly spelled names.

.. _fwhm_doniach:

Approximation to the FWHM of Doniach-Sunjic Line Shape
------------------------------------------------------

The Doniach-Sunjic line shape is commonly used in the analysis of X-ray photoelectron spectroscopy (XPS) data, despite having several limitations. Its popularity in the XPS community stems from its ability to accurately represent asymmetric XPS peaks.

However, there are notable challenges associated with the Doniach-Sunjic line shape:

- The area under this line shape is ill-defined and infinite.
- There is no closed formula available for calculating its full width at half maximum (FWHM).

To approximate the FWHM, the following formula is employed:

.. math::

    \text{FWHM}_{DS} = \gamma \cdot \left(2 + \alpha \cdot a + (\alpha \cdot b)^4\right)

In this formula:

- :math:`\gamma` denotes the broadening parameter of the Doniach-Sunjic line shape.
- :math:`\alpha` represents the asymmetry of the line shape.
- The constants :math:`a = 2.5135` and :math:`b = 3.6398` provide an approximation for the FWHM with an error of less than 2% within a reasonable range of asymmetry, specifically when :math:`\alpha < 0.25`.

When :math:`\alpha = 0`, the Doniach-Sunjic line shape simplifies to a Lorentzian line shape, and thus, this approximation formula corresponds to the FWHM of a Lorentzian as well.
