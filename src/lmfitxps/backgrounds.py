import numpy as np
import copy
import warnings
from functools import lru_cache
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize_scalar
__author__ = "Julian Andreas Hochhaus, Florian Kraushofer"
__copyright__ = "Copyright 2025"
__credits__ = ["Julian Andreas Hochhaus", "Florian Kraushofer"]
__license__ = "MIT"
__version__ = "4.3.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def tougaard_closure():
    """
     .. Hint::
         This function employs a closure to calculate the Tougaard sum only once and subsequently accesses it during subsequent executions of the optimization procedure. When calling `tougaard()`, you are effectively accessing the inner, nested function.

         The concept is as follows:

     .. code-block:: python

         def tougaard_closure():
             bgrnd=[] #store the Tougaard background to optimize performance by avoiding recalculations
             def tougaard_helper():
                 # do actual calculation

         tougaard = tougaard_closure()

     The Tougaard background is based on the four-parameter loss function (4-PIESCS) as suggested by R. Hesse [1]_.

     | In addition to R.Hesse's approach, this model introduces the `extend` parameter, for details, please refer to :ref:`extend_parameter`.

     The Tougaard background is calculated using:

     .. math::

         B_T(E) = \\int_{E}^{\\infty} \\frac{B \\cdot T}{{(C + C_d \\cdot T^2)^2} + D \\cdot T^2} \\cdot y(E') \\, dE'

     where:

         - :math:`B_T(E)` represents the Tougaard background at energy :math:`E`,
         - :math:`y(E')` is the measured intensity at :math:`E'`,
         - :math:`T` is the energy difference :math:`E' - E`.
         - :math:`B` parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. Acts as the scaling factor for the Tougaard background model.
         - :math:`C` , :math:`C_d` and :math:`D` are parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_.

     To generate the 2-PIESCS loss function, set :math:`C_d` to 1 and :math:`D` to 0.
     Set :math:`C_d=1` and :math:`D !=`  :math:`0` to get the 3-PIESCS loss function.

     For further details on the 2-PIESCS loss function, please refer to S.Tougaard [2]_, and for the 3-PIESCS loss function, see S. Tougaard [3]_.


     .. table::
         :widths: auto

         +-----------+---------------+----------------------------------------------------------------------------------------+
         | Parameters|  Type         | Description                                                                            |
         +===========+===============+========================================================================================+
         | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                           |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | B         | :obj:`float`  | B parameter of the 4-PIESCS loss function [1]_.                                        |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | C         | :obj:`float`  | C parameter of the 4-PIESCS loss function [1]_.                                        |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | C_d       | :obj:`float`  | C' parameter of the 4-PIESCS loss function [1]_.                                       |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | D         | :obj:`float`  | D parameter of the 4-PIESCS loss function [1]_.                                        |
         +-----------+---------------+----------------------------------------------------------------------------------------+
         | extend    | :obj:`float`  | Determines, how far the spectrum is extended on the right (in eV). Defaults to 0.      |
         +-----------+---------------+----------------------------------------------------------------------------------------+

     Note
     ----
     This function is used as the model function in the :ref:`TougaardBG` lmfitxps model.
     """
    bgrnd = None

    def tougaard_helper(x, y, B, C, C_d, D, extend=0):
        nonlocal bgrnd
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        extend = int(extend)
        shape = (C, C_d, D, extend)
        if (bgrnd is not None and bgrnd[2] == shape
                and np.array_equal(bgrnd[0], x)
                and np.array_equal(bgrnd[1], y)):
            return B * bgrnd[3]

        delta_x = abs((x[-1] - x[0])) / len(x)
        len_padded = abs(int(extend / delta_x))
        # Continue toward lower BE / higher KE, following the input order.
        signed_step = np.sign(x[-1] - x[0]) * delta_x
        padded_x = np.concatenate([x, x[-1] + signed_step * np.arange(1, len_padded + 1)])
        padded_y = np.concatenate([y, np.full(len_padded, np.mean(y[-10:]))])

        bg = np.zeros_like(x)
        for k, x_k in enumerate(x):
            dx = padded_x[k:] - x_k
            denominator = (C + C_d * dx ** 2) ** 2 + D * dx ** 2
            bg[k] = np.sum(np.abs(dx) / denominator * padded_y[k:] * delta_x)
        # B is only a scale factor; every other integral input belongs in
        # the cache key. Copies also detect in-place edits by callers.
        bgrnd = (x.copy(), y.copy(), shape, bg)
        return B * bg

    return tougaard_helper


# Create the tougaard function with the closure
tougaard = tougaard_closure()

class ShirleyBackgroundWarning(RuntimeWarning):
    """The Shirley coefficient limit prevents reaching a requested endpoint."""


@lru_cache(maxsize=16)
def _shirley_auto_qmax(residual, monotonic):
    """First 20%-of-maximum sensitivity crossing after the global maximum.

    Sensitivity is d(B[0]-const)/d ln(q), evaluated analytically through the
    recurrence. Normalize intensity and orient a negative endpoint step before
    measuring it. ln(alpha) differs from ln(q) by a constant on uniform grids.
    The bounded cache avoids repeating the search when only k/peaks change.
    """
    residual = np.asarray(residual)
    orientation = 1. if residual[0] >= 0 else -1.

    def sensitivity(logq):
        q = np.exp(logq)
        t, w = 1/(1+q), q/(1+q)
        b, derivative = np.zeros_like(q), np.zeros_like(q)
        for value in residual[::-1]:
            excess = value-b
            if monotonic:
                active = excess > 0
                derivative = np.where(active, w*t*excess+t*derivative, derivative)
                b = b+w*np.maximum(excess, 0)
            else:
                derivative = w*t*excess+t*derivative
                b = w*value+t*b
        return orientation*derivative

    # The asymptotic tails lie far beyond one-point to full-window scales.
    grid = np.linspace(np.log(1e-12/max(1, len(residual))), np.log(1e12), 401)
    values = sensitivity(grid)
    peak = int(np.argmax(values))
    if peak == 0 or peak == len(grid)-1 or values[peak] <= 0:
        raise ValueError("Cannot resolve the automatic Shirley sensitivity maximum; "
                         "supply an explicit alpha_max")
    optimum = minimize_scalar(lambda z: -float(sensitivity(z)),
                              bounds=(grid[peak-1], grid[peak+1]), method='bounded',
                              options={'xatol': 1e-10})
    threshold = .2*float(sensitivity(optimum.x))
    left = optimum.x
    for right in grid[grid > left]:
        if sensitivity(right) <= threshold:
            root = brentq(lambda z: float(sensitivity(z))-threshold, left, right,
                          xtol=1e-12)
            return float(np.exp(root))
        left = right
    raise ValueError("Cannot resolve the automatic Shirley slope limit; "
                     "supply an explicit alpha_max")


def shirley(y, k, const, *, x=None, monotonic=False, alpha_max="auto"):
    r"""Calculate the self-consistent Shirley background for XPS spectra.

    For further details, please refer to Shirley [5]_ or Jansson et al. [6]_.
    The normalized background satisfies:

    .. math::

        B_i = c + k(y_0-c)
        \frac{\sum_{j=i}^{N-2}(y_j-B_j)}{\sum_{j=0}^{N-2}(y_j-B_j)}.

    .. table::
        :widths: auto
        :class: parameter-table

        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | Parameters         | Type                      | Description                                                                              |
        +====================+===========================+==========================================================================================+
        | y                  | :obj:`array`              | 1D-array of spectrum intensities.                                                        |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | k                  | :obj:`float`              | Nonnegative dimensionless endpoint scaling. k=0 gives a constant background; k=1         |
        |                    |                           | requests the first data intensity as the first background endpoint.                      |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | const              | :obj:`float`              | Constant background level at the last array point.                                       |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | x                  | :obj:`array`              | Uniformly spaced energies in eV. Required for a numeric alpha_max; optional otherwise.   |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | monotonic          | :obj:`bool`               | Use only positive intensity above the background in the integral. Default False.         |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | alpha_max          | str, float or None        | Coefficient limit: 'auto' (default) uses the 20% slope rule; a positive number sets a    |
        |                    |                           | limit in inverse eV; None disables it.                                                   |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+

    Note
    ----
    The automatic limit is where endpoint sensitivity ``dk/d ln(alpha)``
    falls to 20% of its maximum after that maximum. ``q = alpha * abs(dx)``
    relates the discrete and energy-normalized coefficients.

    The smallest positive root within the limit is selected. If the limit
    prevents reaching the requested endpoint, the limited background is
    returned and k can become insensitive. See :func:`shirley_diagnostics`.
    With no limit, unattainable endpoints raise ValueError.
    """
    background, diagnostics = _shirley_controlled(y, k, const, x, monotonic, alpha_max)
    if diagnostics['cap_active']:
        warnings.warn(
            "Shirley background reached its coefficient limit; the requested "
            "endpoint was not reached and the background may be insensitive to k. "
            "Consider a lower starting k or upper bound; use a smaller alpha_max "
            "if the background consumes excessive peak signal. Inspect "
            "shirley_diagnostics for the effective limit and endpoint mismatch.",
            ShirleyBackgroundWarning, stacklevel=2)
    return background


def shirley_diagnostics(y, k, const, *, x=None, monotonic=False, alpha_max="auto"):
    """Inspect the Shirley coefficient limit and endpoint mismatch.

    Accepts the same arguments as :func:`shirley`. Returns a dictionary:

    .. table:: Diagnostic fields
        :widths: auto
        :class: parameter-table

        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | Field              | Type                      | Description                                                                              |
        +====================+===========================+==========================================================================================+
        | q                  | :obj:`float`              | Effective discrete integral coefficient.                                                 |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | alpha              | float or None             | Effective coefficient in inverse eV; None without x.                                     |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | cap_active         | :obj:`bool`               | Whether the limit prevented reaching the requested endpoint.                             |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | cap_method         | str or None               | 'slope_20_percent', 'explicit', or None for disabled limits.                             |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | q_limit            | float or None             | Selected discrete limit; None if disabled or no calculation was needed.                  |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | alpha_limit        | float or None             | Selected limit in inverse eV; None without x or a calculated limit.                      |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | requested_endpoint | :obj:`float`              | Requested first background value: const + k*(y[0]-const).                                |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
        | actual_endpoint    | float or None             | Calculated first background value; None for empty input.                                 |
        +--------------------+---------------------------+------------------------------------------------------------------------------------------+
    """
    return _shirley_controlled(y, k, const, x, monotonic, alpha_max)[1]


def _shirley_controlled(y, k, const, x, monotonic, alpha_max):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("Shirley intensities must be a finite one-dimensional array")
    if not np.isfinite(k) or k < 0 or not np.isfinite(const):
        raise ValueError("Shirley k must be finite and nonnegative; const must be finite")
    if not isinstance(monotonic, (bool, np.bool_, int, np.integer)) or monotonic not in (0, 1):
        raise ValueError("monotonic must be a boolean")
    automatic = isinstance(alpha_max, str) and alpha_max == 'auto'
    if not automatic and alpha_max is not None:
        if (not isinstance(alpha_max, (int, float, np.number))
                or not np.isfinite(alpha_max) or alpha_max <= 0):
            raise ValueError("alpha_max must be 'auto', positive and finite, or None")
    dx = None
    if x is not None:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape or not np.all(np.isfinite(x)):
            raise ValueError("Shirley energies must be finite and match y")
        if len(x) > 1:
            steps = np.diff(x)
            if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(
                    steps, np.mean(steps), rtol=5e-3, atol=1e-12):
                raise ValueError("Shirley energies must be strictly monotonic and uniformly spaced")
            dx = abs(np.mean(steps))
    if not automatic and alpha_max is not None and x is None:
        raise ValueError("x in eV is required for alpha_max; use alpha_max=None to disable the cap")
    step = k*(y[0]-const) if y.size else 0.
    if monotonic and step < 0:
        raise ValueError("monotonic Shirley requires a nonnegative endpoint step")
    q, capped, qmax = 0., False, None
    background = np.full_like(y, const)
    if y.size > 1 and step != 0:
        scale = max(np.max(np.abs(y[:-1]-const)), abs(step))
        residual = (y[:-1]-const)/scale
        target = step/scale
        if not np.isfinite(scale):
            raise ValueError("Shirley intensity range is not finite")
        def evaluate(q):
            b = np.zeros_like(y)
            fraction = q/(1+q)
            for i in range(len(y)-2, -1, -1):
                excess = residual[i]-b[i+1]
                b[i] = b[i+1] + fraction*(max(excess, 0.) if monotonic else excess)
            return b
        if automatic:
            # Independent of k: a requested endpoint cannot move its own cap.
            cap_residual = y[:-1]-const
            cap_residual = cap_residual/np.max(np.abs(cap_residual))
            qmax = _shirley_auto_qmax(tuple(cap_residual), bool(monotonic))
        else:
            qmax = alpha_max*dx if alpha_max is not None else None
        if monotonic:
            upper = qmax if qmax is not None else 1.
            if qmax is None:
                if target >= max(0., np.max(residual)):
                    raise ValueError("Shirley endpoints have no finite positive-q solution")
                while evaluate(upper)[0] < target:
                    upper *= 2
            if evaluate(upper)[0] < target:
                q, capped = upper, True
            else:
                q = brentq(lambda q: evaluate(q)[0]-target, 0., upper,
                           xtol=5e-324, rtol=4*np.finfo(float).eps)
            background = const + scale*evaluate(q)
        elif qmax is None:
            background = _shirley_uncapped(y, k, const)
            q = step/np.sum(y[:-1]-background[:-1])
        else:
            coefficients = np.trim_zeros(np.r_[residual[0]-target,
                                               np.diff(residual), -residual[-1]], 'f')
            tmin = 1/(1+qmax)
            bracket = _shirley_root_bracket(coefficients, tmin) if coefficients.size else None
            if bracket is None:
                q, capped = qmax, True
            else:
                lower, upper = bracket
                t = lower if lower == upper else brentq(
                    lambda t: np.polynomial.polynomial.polyval(t, coefficients),
                    lower, upper, xtol=5e-324, rtol=4*np.finfo(float).eps)
                q = (1-t)/t
            normalized = evaluate(q)
            if not capped and not np.isclose(normalized[0], target, rtol=1e-8, atol=1e-10):
                raise ValueError("Shirley root failed the self-consistency check")
            background = const + scale*normalized
    return background, dict(q=float(q), alpha=None if dx is None else float(q/dx),
                            cap_active=capped, cap_method='slope_20_percent' if automatic else
                            ('explicit' if alpha_max is not None else None),
                            q_limit=qmax, alpha_limit=None if dx is None or qmax is None else qmax/dx,
                            requested_endpoint=float(const+step),
                            actual_endpoint=float(background[0]) if y.size else None)


def _shirley_uncapped(y, k, const):
    """Solve the original signed equation without a coefficient bound."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("Shirley intensities must be a finite one-dimensional array")
    if not np.isfinite(k) or k < 0 or not np.isfinite(const):
        raise ValueError("Shirley k must be finite and nonnegative; const must be finite")
    step = k * (y[0] - const) if y.size else 0.0
    if y.size < 2 or step == 0:
        return np.full_like(y, const)

    # Work relative to the baseline and scale before solving, so tolerance
    # does not depend on the intensity units or a large constant offset.
    residual = y[:-1] - const
    scale = max(np.max(np.abs(residual)), abs(step))
    residual = residual / scale
    target = step / scale
    if not np.isfinite(scale) or not np.all(np.isfinite(residual)):
        raise ValueError("Shirley intensity range is not finite")

    # With t = 1/(1+q), b_i = (1-t)*residual_i + t*b_(i+1).
    # The endpoint equation is a polynomial on 0 < t < 1. At k=1,
    # t=0 is an infinite-q, zero-integral limit, NOT a valid solution.
    coefficients = np.r_[residual[0] - target,
                         np.diff(residual), -residual[-1]]
    coefficients = np.trim_zeros(coefficients, trim='f')
    if not coefficients.size:
        raise ValueError("Shirley endpoint equation is degenerate")

    def endpoint(t):
        return np.polynomial.polynomial.polyval(t, coefficients)

    lower, upper = _shirley_root_bracket(coefficients)
    t = lower if lower == upper else brentq(
        endpoint, lower, upper, xtol=5e-324, rtol=4*np.finfo(float).eps)
    if not 0 < t < 1:
        raise ValueError("Shirley endpoints have no finite positive-q solution")
    derivative = np.arange(1, len(coefficients))*coefficients[1:]
    slope = np.polynomial.polynomial.polyval(t, derivative)
    slope_scale = np.polynomial.polynomial.polyval(t, np.abs(derivative))
    if abs(slope) <= 8*np.sqrt(np.finfo(float).eps)*slope_scale:
        raise ValueError("Shirley root is multiple or too ill-conditioned to resolve")
    background = np.zeros_like(y)
    for i in range(y.size - 2, -1, -1):
        background[i] = (1-t)*residual[i] + t*background[i+1]

    cumulative = np.r_[np.cumsum((residual-background[:-1])[::-1])[::-1], 0.0]
    total = cumulative[0]
    if total == 0 or not np.allclose(
            background, target*cumulative/total, rtol=1e-8, atol=1e-10):
        raise ValueError("Shirley root failed the self-consistency check")
    return const + scale*background


def _shirley_root_bracket(coefficients, minimum=0.0):
    """Isolate the largest root in (0, 1), using Bernstein sign bounds.

    Largest t means smallest positive q: the branch reached first from
    the constant-background limit. Subdivision avoids skipping two roots
    with equal signs at a coarse bracket's endpoints. Ambiguous/tangent
    roots raise rather than silently choosing another branch.
    """
    degree = len(coefficients) - 1
    indices = np.arange(degree + 1)
    weights = np.ones(degree + 1)
    bernstein = np.full(degree + 1, coefficients[0])
    for i in range(1, degree + 1):
        weights[:i] = 0
        weights[i:] *= (indices[i:] - i + 1) / (degree - i + 1)
        bernstein += coefficients[i] * weights
    # Use directly evaluated endpoints to avoid conversion roundoff.
    bernstein[-1] = np.polynomial.polynomial.polyval(1.0, coefficients)
    if minimum:
        work = bernstein.copy()
        right = np.empty_like(work)
        right[-1] = work[-1]
        for i in range(1, len(work)):
            work = (1-minimum)*work[:-1] + minimum*work[1:]
            right[-i-1] = work[-1]
        bernstein = right
        bernstein[0] = np.polynomial.polynomial.polyval(minimum, coefficients)
    stack = [(minimum, 1.0, bernstein, 0)]
    while stack:
        lower, upper, values, depth = stack.pop()
        nonzero = values[values != 0]
        variations = np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1]))
        if variations == 0:
            if 0 < upper < 1 and values[-1] == 0:
                return upper, upper
            if 0 < lower < 1 and values[0] == 0:
                return lower, lower
            continue
        if (variations == 1 and values[0] != 0 and values[-1] != 0
                and np.signbit(values[0]) != np.signbit(values[-1])):
            return lower, upper
        if depth >= 48:
            raise ValueError("Shirley root is multiple or cannot be reliably isolated")
        work = values.copy()
        left, right = np.empty_like(values), np.empty_like(values)
        left[0], right[-1] = work[0], work[-1]
        for i in range(1, len(values)):
            work = (work[:-1] + work[1:]) / 2
            left[i], right[-i-1] = work[0], work[-1]
        middle = (lower + upper) / 2
        stack.append((lower, middle, left, depth+1))
        stack.append((middle, upper, right, depth+1))
    if minimum:
        return None
    raise ValueError("Shirley endpoints have no finite positive-q solution")

def slope(y, k):
    """
    Calculates the Slope background for X-ray photoelectron spectroscopy (XPS) spectra.
    The Slope Background is implemented as suggested by A. Herrera-Gomez et al in [4]_.
    Hereby, while the Shirley background is designed to account for the difference in background height between the two sides of a peak, the Slope background is designed to account for the change in slope.
    This is done in a manner that resembles the Shirley method:

    .. math::

        \\frac{B_{\\text{Slope}}(E)}{dE} = -k_{\\text{Slope}} \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} ] \\, dE'

    where:

        - :math:`\\frac{B_{\\text{Slope}}(E)}{dE}` represents the slope of the background at energy :math:`E`,
        - :math:`I(E')` is the measured intensity at :math:`E'`,
        - :math:`I_{\\text{right}}` is the measured intensity of the rightmost datapoint,
        - :math:`k_{\\text{Slope}}` parameter to scale the integral to resemble the measured data. This parameter is related to the Tougaard background. For details see [4]_.

    To get the background itself, equation :math:numref:`slope` is integrated:

    .. math::

         B_{\\text{Slope}}(E)= \\int_{E}^{E_{\\text{right}}} [\\frac{B_{\\text{Slope}}(E')}{dE'}] \\, dE'


    .. table::
       :widths: auto

       +-----------+---------------+----------------------------------------------------------------------------------------+
       | Parameters|  Type         | Description                                                                            |
       +===========+===============+========================================================================================+
       | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                        |
       +-----------+---------------+----------------------------------------------------------------------------------------+
       | k         | :obj:`float`  | Slope parameter :math:`k_{\\text{Slope}}`.                                              |
       +-----------+---------------+----------------------------------------------------------------------------------------+

    Note
    ----
    This function is used as the model function in the :ref:`SlopeBG` lmfitxps model

    Warning
    -------
    Please note that the Slope background should not be solely relied upon to mimic a measured XPS background. It is advisable to use it combined with other background functions, such as the Shirley background.
    For further details, please refer to A. Herrera-Gomez et al [4]_.
    """
    n = len(y)
    y_right = np.min(y)
    y_temp = y - y_right
    temp = np.cumsum(y_temp[::-1])[::-1]
    bg = -np.cumsum(temp[::-1])[::-1]

    return -k * bg


def shirley_calculate(x, y, tol=1e-5, maxit=10, bounds=None):
    """
    Calculates the Shirley background for a given set of x (energy) and y (intensity) data.

    The implementation was inspired by the python implementation of Kane O'Donnell [5]_.

    The Shirley background is calculated iteratively:

    .. math::
        :label: shirleystatic

        B_{S, n}(E) = k_n \\cdot \\int_{E}^{E_{\\text{right}}} [I(E') - I_{\\text{right}} - B_{S, n-1}(E')] \\, dE'


    where:
        - :math:`B_{S, n}(E)` represents the Shirley background at :math:`E` in the :math:`n`-th iteration,
        - :math:`I(E')` is the intensity at :math:`E'`,
        - :math:`k_n` is the Shirley scaling parameter for the :math:`n`-th iteration.
        - :math:`E_{\\text{right}}` and :math:`I_{\\text{right}}` are the rightmost energy/intensity of the dataset.

    The iterative process continues until the difference :math:`B_{S, n}(E) - B_{S, n-1}(E)` is suitable small or the number of maximum iterations :math:`maxit` is exceeded.

    Initially, :math:`B_{S, 0}(E)=0` is chosen and :math:`k_n` is found from the requirement that :math:`\\left(I_{\\text{left}}-B_{S, n}(E_{\\text{left}})\\right)=0`.
    For further details, please refer to e.g. S. Tougaard [6]_ .

    Typically, convergence is reached after :math:`\\approx 5` iterations. The convergence criterion is:

     .. math::
        :label: shirleyconvergence

        \\langle\\left(B_{S, n}(E)-B_{S, n-1}(E)\\right)^2\\rangle<tol


    Parameters:
    -----------

    .. table:: Available parameters
        :widths: auto
        :class: parameter-table

        +--------------+------------------+------------------------------------------------------------------------------------------+
        | Parameter    | Type             | Description                                                                              |
        +==============+==================+==========================================================================================+
        | x            | :obj:`array`     | 1D-array containing the x-values (energies) of the spectrum.                             |
        +--------------+------------------+------------------------------------------------------------------------------------------+
        | y            | :obj:`array`     | 1D-array containing the y-values (intensities) of the spectrum.                          |
        +--------------+------------------+------------------------------------------------------------------------------------------+
        | tol          | :obj:`float`     | Tolerance for the mean squared change between iterations. Defaults to 1e-5.              |
        +--------------+------------------+------------------------------------------------------------------------------------------+
        | maxit        | :obj:`int`       | Maximum number of iterations before calculation is interrupted. Defaults to 10.          |
        +--------------+------------------+------------------------------------------------------------------------------------------+
        | bounds       | :obj:`tuple`     | Either two x values or two (x,y) pairs. Determines the edges of the Shirley background.  |
        |              |                  | Background will be constant outside this range. If only x is passed, picks the y of      |
        |              |                  | closest data point. If nothing is passed, uses the edges of the data range.              |
        +--------------+------------------+------------------------------------------------------------------------------------------+

    Returns:
    --------
        :obj:`array`:  The function returns the calculated Shirley background as an :obj:`array`.

    Hint
    ----

    This function retains trapezoidal fixed-point iteration and the optional
    endpoint bounds. If maxit is reached without convergence, it prints a
    message and returns the last iterate.

    For a static calculation using the root solver, call :func:`shirley`
    directly with ``k=1`` and ``const=y[-1]``. That function also supports
    ``monotonic`` and ``alpha_max``, but uses a different discretization and
    does not handle bounds. For background parameters that vary during peak
    fitting, use :ref:`ShirleyBG`.

    """

    # Sanity check: Do we actually have data to process here?
    if not (any(x) and any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return x * 0            # TODO: raise ValueError instead?
    if not len(x) == len(y):
        print("Length missmatch between x and y. Returning zero background")
        return x * 0            # TODO: raise ValueError instead?

    # couple x and y values for easier handling in the following;
    #  data will be modified in-place, but this keeps the input x,y safe.
    data = np.array((x, y))

    if not bounds:
        bounds = np.array((data[:, 0], data[:, -1])).T
    else:
        bounds = np.array(bounds).T
        if bounds.shape == (2,):
            # bounds are only energies, don't have values yet.
            # cut the range, then use closest data values.
            data = data[:, (data[0] >= np.min(bounds)) &
                           (data[0] <= np.max(bounds))]
            bounds = np.array((data[:,0], data[:,-1])).T
        else:
            # if bounds are not at the ends of the data,
            # consider only the inner parts of the data from here on
            data = data[:, (data[0] >= np.min(bounds[0])) &
                           (data[0] <= np.max(bounds[0]))]
            # make sure that bounds are actually part of the x range: 
            # keep their y values, put x on the closest existing point
            bounds[0, 0] = data[0, 0]
            bounds[0, 1] = data[0, -1]

    # ensure that the 'left' value of the data is higher than the 'right'
    # NOTE: This is insensitive to whether the energy axis is binding or
    # kinetic, but WILL give unphysical results where the background goes
    # 'up' without complaining if that's what's in the data!
    if data[1, 0] < data[1, -1]:
        is_reversed = True
        data = data[:,::-1]
    else:
        is_reversed = False
    # make the bounds follow the same order as the data;
    # i.e. if kinetic energy -> lower value first, otherwise higher first
    if (np.sign(bounds[0, 0] - bounds[0, -1])
            != np.sign(data[0, 0] - data[0, -1])):
        bounds = bounds[:, ::-1]

    # Initial value of the background shape B. The total background S = bounds[1,1] + B,
    # and B is initially zero
    B = data[1] * 0

    for it in range(maxit):
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        # background-subtracted y so far, and cumulative integral:
        y_sub = data[1] - B - bounds[1, 1]
        y_int = cumulative_trapezoid(y_sub[::-1], data[0, ::-1], initial=0)[::-1]
        # Calculate new k = (yl - yr) / (integral of y over the whole range)
        k = (bounds[1, 0] - bounds[1, 1]) / y_int[0]
        # new B is simply the cumulative integral normalized by the new k
        B_new = k*y_int
        # If B_new is close to B, exit.
        if np.sum((B - B_new)**2) / len(B) < tol:
            B = np.copy(B_new)
            break
        else:
            B = np.copy(B_new)
    else:
        print("Max iterations exceeded before convergence.")
    B += bounds[1, 1]
    if is_reversed:
        B = B[::-1]
        data = data[:,::-1]

    # check the original data range, fill up the missing parts
    npx = np.array(x)
    index_exists = np.where((npx >= np.min(data[0])) & 
                            (npx <= np.max(data[0])))[0]
    B_whole_range = np.concatenate((
        np.full(index_exists[0], B[0]),
        B,
        np.full(len(npx) - index_exists[-1] - 1, B[-1])
        ))
    return B_whole_range


def tougaard_calculate(x, y, tb=2866, tc=1643, tcd=1, td=1, maxit=100):
    """
    Calculates the Tougaard background for a given set of x and y data. 
    The calculation is hereby based on the four-parameter loss function (4-PIESCS) as suggested by R.Hesse [1]_.

    The implementation was inspired by the IGOR implementation of James Mudd [2]_.

    The Tougaard background is calculated using:

    .. math::
        :label: tougaard

        B_T(E) = \\int_{E}^{\\infty} \\frac{B \\cdot T}{{(C + C_d \\cdot T^2)^2} + D \\cdot T^2} \\cdot y(E') \\, dE'

    where:

        - :math:`B_T(E)` represents the Tougaard background at energy :math:`E`,
        - :math:`y(E')` is the measured intensity at :math:`E'`,
        - :math:`T` is the energy difference :math:`E' - E`.
        - :math:`B` parameter of the 4-PIESCS loss function as introduced by R. Hesse [1]_. Acts as the scaling factor for the Tougaard background model. This parameter is the only one varied during the calculation.
        - :math:`C` , :math:`C_d` and :math:`D` are parameter of the 4-PIESCS loss function as introduced by R.Hesse [1]_. These parameters are kept fixed during the calculation.

    To generate the 2-PIESCS loss function, set :math:`C_d` to 1 and :math:`D` to 0.
    Set :math:`C_d=1` and :math:`D !=`  :math:`0` to get the 3-PIESCS loss function.

    For further details on the 2-PIESCS loss function, please refer to S.Tougaard [3]_, and for the
    3-PIESCS loss function, see S. Tougaard [4]_.

    During the calculation, the Tougaard background is calculated using the provided start parameters based on equation :math:numref:`tougaard`. This process is iteratively repeated, adapting the :math:`B` parameter, until convergence is reached or the number of iterations exceeds :math:`maxit`.

    The convergence is hereby defined by the deviation between the calculated Tougaard background :math:`B_T(E)` and the measured intensity :math:`y(E)` at the leftmost datapoint.

    The Tougaard background is considered to converge if :math:`|B_T(E)-y(E)|< 10^{-6}\\cdot B_T(E)` is fulfilled.

    Parameters:
    -----------

    .. table:: Available parameters
        :widths: auto

        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | Parameter |  Type         | Description                                                                                                                                    |
        +===========+===============+================================================================================================================================================+
        | x         | :obj:`array`  | 1D-array containing the x-values (energies) of the spectrum.                                                                                   |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | y         | :obj:`array`  | 1D-array containing the y-values (intensities) of the spectrum.                                                                                |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tb        | :obj:`float`  | B parameter of the 4-PIESCS loss function [1]_. Acts as scaling parameter and is optimized during the fit. Defaults to 2866 as starting value. |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tc        | :obj:`float`  | C parameter of the 4-PIESCS loss function [1]_. Defaults to 1643.                                                                              |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | tcd       | :obj:`float`  | C' parameter of the 4-PIESCS loss function [1]_. Defaults to 1.                                                                                |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | td        | :obj:`float`  | D parameter of the 4-PIESCS loss function [1]_. Defaults to 1.                                                                                 |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+
        | maxit     | :obj:`int`    | Maximum number of iterations before calculation is interrupted. Defaults to 100.                                                               |
        +-----------+---------------+------------------------------------------------------------------------------------------------------------------------------------------------+

    Returns:
    --------
        :obj:`tuple` of (:obj:`numpy.ndarray`, :obj:`float`):  The function returns a tuple consisting of the calculated Tougaard background as a :obj:`numpy.array` and the Tougaard scale parameter :math:`B` as :obj:`float`.

    Hint
    ----

    This function should be used, if you intend to calculate and remove the background from your data before starting the fitting procedure, if you instead wish to include the background in the fitting model, please use the desired background model, e.g. :ref:`TougaardBG`.

    """
    # Sanity check: Do we actually have data to process here?
    if not (np.any(x) and np.any(y)):
        print("One of the arrays x or y is empty. Returning zero background.")
        return [np.asarray(x * 0), tb]

    # KE in XPS or PE in XAS
    if x[0] < x[-1]:
        is_reversed = True
    # BE in XPS
    else:
        is_reversed = False

    Btou = y * 0

    it = 0
    while it < maxit:
        if not is_reversed:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (y[j] - y[len(y) - 1]) * (x[0] - x[1]) * (x[i] - x[j]) / (
                            (tc + tcd * (x[i] - x[j]) ** 2) ** 2 + td * (x[i] - x[j]) ** 2)
                Btou[i] = Bint * tb

        else:
            for i in range(len(y) - 1, -1, -1):
                Bint = 0
                for j in range(len(y) - 1, i - 1, -1):
                    Bint += (y[j] - y[len(y) - 1]) * (x[1] - x[0]) * (x[j] - x[i]) / (
                            (tc + tcd * (x[j] - x[i]) ** 2) ** 2 + td * (x[j] - x[i]) ** 2)
                Btou[i] = Bint * tb

        Boffset = Btou[0] - (y[0] - y[len(y) - 1])
        if abs(Boffset) < (0.000001 * Btou[0]) or maxit == 1:
            break
        else:
            tb = tb - (Boffset / Btou[0]) * tb * 0.5
        it += 1

    print("Tougaard B:", tb, ", C:", tc, ", C':", tcd, ", D:", td)

    return np.asarray(y[len(y) - 1] + Btou), tb
