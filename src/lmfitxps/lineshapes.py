import warnings

import numpy as np
from lmfit.lineshapes import doniach, gaussian, thermal_distribution
from scipy.signal import convolve as sc_convolve
__author__ = "Julian Andreas Hochhaus"
__copyright__ = "Copyright 2023"
__credits__ = ["Julian Andreas Hochhaus"]
__license__ = "MIT"
__version__ = "4.2.0"
__maintainer__ = "Julian Andreas Hochhaus"
__email__ = "julian.hochhaus@tu-dortmund.de"

def _dublett_oversampling(x, sigma, fct_coster_kronig, max_oversampling):
    """Return the required and capped internal-grid oversampling factors."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional array with at least two points")
    steps = np.diff(x)
    if np.any(steps == 0) or not np.all(np.sign(steps) == np.sign(steps[0])):
        raise ValueError("x must be strictly monotonic")

    absolute_steps = np.abs(steps)
    data_step = np.median(absolute_steps)
    if not np.allclose(absolute_steps, data_step, rtol=0.01, atol=1e-12):
        raise ValueError("x must be uniformly spaced")
    narrowest_sigma = sigma * min(1.0, fct_coster_kronig)
    if narrowest_sigma <= 0:
        required = max_oversampling
    else:
        # Aim for four samples across the narrowest intrinsic width.
        ratio = 4 * data_step / narrowest_sigma
        required = max(1, int(np.ceil(ratio - 1e-12)))
    return required, min(required, max_oversampling)


def dublett_components(
        x, amplitude, sigma, gamma, gaussian_sigma, center, soc,
        height_ratio, fct_coster_kronig, max_oversampling=10):
    """Evaluate the two convolved dublett components separately."""
    x = np.asarray(x, dtype=float)
    if not isinstance(max_oversampling, (int, np.integer)) or max_oversampling < 1:
        raise ValueError("max_oversampling must be a positive integer")

    required, oversampling = _dublett_oversampling(
        x, sigma, fct_coster_kronig, max_oversampling
    )
    if required > max_oversampling:
        warnings.warn(
            "The intrinsic dublett width requires a finer internal grid than "
            "the configured max_oversampling permits. The convolved profile "
            "may depend on grid alignment; increase max_oversampling "
            "deliberately if needed.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_internal = (x.size - 1) * oversampling + 1
    x_internal = np.linspace(x[0], x[-1], n_internal)
    is_binding_energy = x_internal[-1] < x_internal[0]
    second_center = center + soc if is_binding_energy else center - soc
    kernel = (
        1 / (np.sqrt(2 * np.pi) * gaussian_sigma)
        * gaussian(
            x_internal,
            amplitude=1,
            center=np.mean(x_internal),
            sigma=gaussian_sigma,
        )
    )

    # Reflect each intrinsic profile about its own center for binding energy;
    # reversing the symmetric Gaussian kernel cannot reverse the Doniach tail.
    primary = fft_convolve(
        doniach(
            2 * center - x_internal if is_binding_energy else x_internal,
            amplitude=1, center=center, sigma=sigma, gamma=gamma
        ),
        kernel,
        is_binding_energy=is_binding_energy,
    )
    secondary = fft_convolve(
        doniach(
            2 * second_center - x_internal if is_binding_energy else x_internal,
            amplitude=height_ratio,
            center=second_center,
            sigma=fct_coster_kronig * sigma,
            gamma=gamma,
        ),
        kernel,
        is_binding_energy=is_binding_energy,
    )
    if is_binding_energy:
        primary = np.interp(x[::-1], x_internal[::-1], primary[::-1])[::-1]
        secondary = np.interp(x[::-1], x_internal[::-1], secondary[::-1])[::-1]
    else:
        primary = np.interp(x, x_internal, primary)
        secondary = np.interp(x, x_internal, secondary)

    # Retain the existing public meaning of amplitude as the maximum on the
    # supplied x grid, while applying one common scale factor to both peaks.
    scale = amplitude / np.max(primary + secondary)
    return primary * scale, secondary * scale


def dublett(
        x, amplitude, sigma, gamma, gaussian_sigma, center, soc,
        height_ratio, fct_coster_kronig, max_oversampling=10):
    """
    Calculates the convolution of a Doniach-Sunjic Dublett with a Gaussian.

    The intrinsic profiles and convolution are evaluated on an adaptively
    oversampled grid before interpolation onto the input grid. The public
    height_ratio name is retained for compatibility, but represents the
    requested intrinsic amplitude/area ratio.

    Parameters
    ----------
    x: array-like
        Energy array: descending binding energy or ascending kinetic energy.
    amplitude: float
        Maximum amplitude of the combined convolved profile.
    sigma: float
        Sigma of the primary Doniach profile.
    gamma: float
        Asymmetry factor of both Doniach profiles. Positive values give a
        tail toward higher binding energy (lower kinetic energy).
    gaussian_sigma: float
        Sigma of the Gaussian convolution kernel.
    center: float
        Center of the primary peak.
    soc: float
        Absolute spin-orbit separation.
    height_ratio: float
        Requested intrinsic amplitude/area ratio of secondary to primary.
    fct_coster_kronig: float
        Ratio of secondary to primary Doniach sigma.
    max_oversampling: int, optional
        Maximum internal-grid oversampling factor. Defaults to 10.

    Returns
    -------
    array-like
        Convolution of the Doniach dublett and Gaussian profile.
    """
    primary, secondary = dublett_components(
        x, amplitude, sigma, gamma, gaussian_sigma, center, soc,
        height_ratio, fct_coster_kronig, max_oversampling=max_oversampling
    )
    return primary + secondary


def singlett(x, amplitude, sigma, gamma, gaussian_sigma, center):
    """
    Calculates the convolution of a Doniach-Sunjic with a Gaussian.
    Thereby, the Gaussian acts as the convolution kernel.

    Parameters
    ----------
    x: array-like
        Energy array: descending binding energy or ascending kinetic energy.
    amplitude: float
        factor used to scale the calculated convolution to the measured spectrum. This factor is used as
        the amplitude of the Doniach profile.
    sigma: float
        Sigma of the Doniach profile
    gamma: float
        Asymmetry factor of the Doniach profile. Positive values give a
        tail toward higher binding energy (lower kinetic energy).
    gaussian_sigma: float
        sigma of the gaussian profile which is used as the convolution kernel
    center: float
        position of the maximum of the measured spectrum

    Returns
    ---------
    array-type
        convolution of a doniach profile and a gaussian profile
    """
    x = np.asarray(x, dtype=float)
    is_binding_energy= x[-1] < x[0]
    # lmfit's Doniach tail points toward lower numerical energy. Reflect
    # the profile about its center on the binding-energy scale.
    doniach_x = 2 * center - x if is_binding_energy else x
    conv_temp = fft_convolve(doniach(doniach_x, amplitude=1, center=center, sigma=sigma, gamma=gamma),
                                 1 / (np.sqrt(2 * np.pi) * gaussian_sigma) * gaussian(x, amplitude=1, center=np.mean(x),
                                                                                      sigma=gaussian_sigma), is_binding_energy=is_binding_energy)
    return amplitude * conv_temp / max(conv_temp)


kb = 8.6173e-5  # Boltzmann k in eV/K , replace by scipy const value


def fermi_edge(x, amplitude, center, kt, sigma):
    """
    Calculates the convolution of a Thermal Distribution (Fermi-Dirac Distribution) with a Gaussian.
    Thereby, the Gaussian acts as the convolution kernel.

    Parameters
    ----------
    x: array-like
        Array containing the energy of the spectrum to fit. Works for both, kinetic+binding energy scaled data.
    amplitude: float
        factor used to scale the calculated convolution to the measured spectrum. This factor is used
        as the amplitude of the Gaussian Kernel.
    center: float
        position of the step of the fermi edge
    kt: float
        boltzmann constant in eV multiplied with the temperature T in kelvin
        (i.e. for room temperature kt=kb*T=8.6173e-5 eV/K*300K=0.02585 eV)
    sigma: float
        Sigma of the gaussian profile which is used as the convolution kernel



    Returns
    ---------
    array-type
        convolution of a fermi dirac distribution and a gaussian profile
    """
    is_binding_energy= x[-1] < x[0]

    if is_binding_energy:
        kt=-kt
    conv_temp = fft_convolve(thermal_distribution(x, amplitude=1, center=center, kt=kt, form='fermi'),
                                 1 / (np.sqrt(2 * np.pi) * sigma) * gaussian(x, amplitude=1, center=np.mean(x),
                                                                             sigma=sigma), is_binding_energy=is_binding_energy)
    return amplitude * conv_temp / max(conv_temp)


def convolve(data, kernel, is_binding_energy=False):
    """
    Calculates the convolution of a data array with a kernel by using numpy convolve function.
    To suppress edge effects and generate a valid convolution on the full data range, the input dataset is extended
    at the edges.

    Parameters
    ----------
    data: array-like
        1D-array containing the data to convolve
    kernel: array-like
        1D-array which defines the kernel used for convolution. If binding energy scale is used, the kernel is inverted/flipped.
    is_binding_energy: boolean
        Boolean determining type of energy scale which determines the orientation of the kernel

    Returns
    ---------
    array-type
        convolution of a data array with a kernel array

    See Also
    ---------
    numpy.convolve()
    """
    if is_binding_energy:
        kernel=kernel[::-1]
    min_num_pts = min(len(data), len(kernel))
    padding = np.ones(min_num_pts)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    out = np.convolve(padded_data, kernel, mode='valid')
    n_start_data = int((len(out) - min_num_pts) / 2)
    return (out[n_start_data:])[:min_num_pts]


def fft_convolve(data, kernel, is_binding_energy=False):
    """
    Calculates the convolution of a data array with a kernel by using the convolution theorem and thereby
    transforming the time-consuming convolution operation into a multiplication of FFTs.
    The convolution using this approach is done using the `scipy.signal.convolve()` function with the `method="fft"` attribute.
    To suppress edge effects and generate a valid convolution on the full data range, the input dataset is
    extended at the edges.

    Parameters
    ----------
    data: array-like
        1D-array containing the data to convolve
    kernel: array-like
        1D-array which defines the kernel used for convolution. If binding energy scale is used, the kernel is inverted/flipped.
    is_binding_energy: boolean
        Boolean determining type of energy scale which determines the orientation of the kernel

    Returns
    ---------
    array-type
        convolution of a data array with a kernel array

    See Also
    ---------
    scipy.signal.convolve()
    """
    if is_binding_energy:
        kernel=kernel[::-1]
    min_num_pts = min(len(data), len(kernel))
    padding = np.ones(min_num_pts)
    padded_data = np.concatenate((padding * data[0], data, padding * data[-1]))
    out = sc_convolve(padded_data, kernel, mode='valid', method="fft")
    n_start_data = int((len(out) - min_num_pts) / 2)
    return (out[n_start_data:])[:min_num_pts]

