Static backgrounds
==================
.. index:: backgrounds


Static backgrounds are calculated separately from peak fitting. They can be
subtracted before fitting or integrating the remaining signal. In contrast,
the models in :ref:`BGModels` allow background parameters to vary during a fit.

The functions below use iterative calculations. The root-based Shirley
function can also be evaluated directly, without an lmfit model.

.. _shirley_calculate:

:py:func:`shirley_calculate`
____________________________


.. autofunction:: lmfitxps.backgrounds.shirley_calculate

Using the root solver without fitting
-------------------------------------

``shirley_calculate`` retains its explicit iterative algorithm, including
``tol``, ``maxit`` and the optional endpoint ``bounds``. To use the root solver
and its coefficient-limit and monotonic options instead::

    from lmfitxps.backgrounds import shirley

    # Uniformly spaced x, with the low-binding-energy endpoint last.
    background = shirley(y, k=1, const=y[-1], x=x)
    signal = y - background

This requests both data endpoints; the default automatic limit may prevent
reaching the first one. Set ``alpha_max=None`` to require an uncapped root
(an unattainable endpoint then raises ValueError), or ``monotonic=True`` to
exclude negative contributions to the background integral.

The two functions use different discretizations: ``shirley_calculate`` uses
trapezoidal integration, whereas ``shirley`` uses a uniform-grid rectangle
sum. Their results therefore need not match exactly. The direct root function
does not provide the ``bounds`` handling of ``shirley_calculate``.

.. _tougaard_calculate:

:py:func:`tougaard_calculate`
_____________________________

.. autofunction:: lmfitxps.backgrounds.tougaard_calculate
.. _lmfit.model.Model: https://lmfit.github.io/lmfit-py/model.html#


References
__________
.. [1] Hesse, R., Denecke, R. (2011). Improved Tougaard background calculation by introduction of fittable parameters for the inelastic electron scattering cross-section in the peak fit of photoelectron spectra with UNIFIT 2011.,43(12), 1514–1526. https://doi.org/10.1002/sia.3746
.. [2] Mudd, J. (2011). Igor procedure for subtracting XPS backgrounds. https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/surface/people/james_mudd/igor/
.. [3] Tougaard, S. (1987). Low energy inelastic electron scattering properties of noble and transition metals. Solid State Communications, 61(9), 547–549. https://doi.org/10.1016/0038-1098(87)90166-9
.. [4] Tougaard, S. (1997). Universality Classes of Inelastic Electron Scattering Cross-sections. Surf. Interface Anal., 25: 137-154. https://doi.org/10.1002/(SICI)1096-9918(199703)25:3<137::AID-SIA230>3.0.CO;2-L
.. [5] O'Donnell, K., (2013) Implementation of the auto-Shirley background. https://github.com/kaneod/physics/blob/master/python/specs.py
.. [6] Tougaard, S. (2021). Practical guide to the use of backgrounds in quantitative XPS. Journal of Vacuum Science & Technology A; 39 (1): 011201. https://doi.org/10.1116/6.0000661
