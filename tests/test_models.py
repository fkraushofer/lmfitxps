import numpy as np
import lmfit
from src.lmfitxps import models
import pytest


@pytest.fixture()
def dublett_model():
    """Return a Doniach-Sunjic convolved Dublett peak model."""
    return models.ConvGaussianDoniachDublett(prefix='peak_')


@pytest.fixture()
def tougaard_model():
    """Return a Tougaard background model."""
    return models.TougaardBG(prefix='tougaard_', independent_vars=['x', 'y'])


def test_fit_tougaard_dublett(tougaard_model, dublett_model):
    data = np.genfromtxt('examples/clean_Au_4f.csv', delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]

    params = lmfit.Parameters()

    params.add('tougaard_B', value=200)
    params.add('tougaard_C', value=144.506, vary=False)
    params.add('tougaard_C_d', value=0.281, vary=False)
    params.add('tougaard_D', value=268.598, vary=False)
    params.add('tougaard_extend', value=50, vary=False)
    params.add('peak_amplitude', value=80000, min=0)
    params.add('peak_sigma', value=0.2, min=0)
    params.add('peak_gamma', value=0.02)
    params.add('peak_gaussian_sigma', value=0.2, min=0)
    params.add('peak_center', value=92)
    params.add('peak_soc', value=3.67, vary=False)
    params.add('peak_height_ratio', value=0.75, min=0)
    params.add('peak_fct_coster_kronig', value=1, min=0)
    fit_model = tougaard_model + dublett_model
    result = fit_model.fit(y, params, y=y, x=x)
    assert result.success
    assert result.errorbars
    ratio_report = models.dublett_ratio_report(result)
    assert "requested area ratio" in ratio_report
    assert "sampled area ratio" in ratio_report


def test_dublett_preserves_ratio_for_narrow_intrinsic_peaks(dublett_model):
    """The doublet ratio must not depend on alignment with the x grid."""
    x = np.linspace(80.0, 68.0, 121)
    center = 71.25
    soc = 3.33
    height_ratio = 0.75

    params = dublett_model.make_params(
        amplitude=1.0,
        sigma=0.01,
        gamma=0.0,
        gaussian_sigma=0.67,
        center=center,
        soc=soc,
        height_ratio=height_ratio,
        fct_coster_kronig=1.0,
    )
    with pytest.warns(RuntimeWarning, match="max_oversampling"):
        doublet = dublett_model.eval(params, x=x)

    primary_peak = np.max(doublet[np.abs(x - center) < 1.0])
    secondary_peak = np.max(doublet[np.abs(x - (center + soc)) < 1.0])

    assert secondary_peak / primary_peak == pytest.approx(
        height_ratio, rel=0.02
    )



def test_dublett_reports_actual_ratios_and_oversampling(dublett_model):
    """Diagnostics describe the profiles actually sampled on the fit grid."""
    x = np.linspace(80.0, 68.0, 121)
    params = dublett_model.make_params(
        amplitude=1.0,
        sigma=0.01,
        gamma=0.0,
        gaussian_sigma=0.67,
        center=71.25,
        soc=3.33,
        height_ratio=0.75,
        fct_coster_kronig=1.0,
    )

    with pytest.warns(RuntimeWarning, match="max_oversampling"):
        diagnostics = dublett_model.ratio_diagnostics(params, x)

    assert diagnostics["requested_area_ratio"] == 0.75
    assert diagnostics["sampled_area_ratio"] == pytest.approx(0.75, rel=0.02)
    assert diagnostics["sampled_height_ratio"] == pytest.approx(0.75, rel=0.02)
    assert diagnostics["required_oversampling"] == 40
    assert diagnostics["used_oversampling"] == 10
    assert diagnostics["oversampling_limit_reached"]


def test_dublett_accepts_user_defined_oversampling_limit():
    """Users can consciously allow a finer internal grid."""
    model = models.ConvGaussianDoniachDublett(
        prefix="peak_", max_oversampling=25
    )
    x = np.linspace(80.0, 68.0, 121)
    params = model.make_params(
        amplitude=1.0,
        sigma=0.01,
        gamma=0.0,
        gaussian_sigma=0.67,
        center=71.25,
        soc=3.33,
        height_ratio=0.75,
        fct_coster_kronig=1.0,
    )

    with pytest.warns(RuntimeWarning, match="max_oversampling"):
        diagnostics = model.ratio_diagnostics(params, x)

    assert diagnostics["used_oversampling"] == 25
    assert diagnostics["required_oversampling"] == 40



def test_dublett_accepts_rounded_uniform_grid(dublett_model):
    """Typical decimal rounding in exported energy grids is acceptable."""
    x = np.round(np.linspace(80.0, 68.0, 294), 4)
    params = dublett_model.make_params(
        amplitude=1.0,
        sigma=0.2,
        gamma=0.0,
        gaussian_sigma=0.67,
        center=71.25,
        soc=3.33,
        height_ratio=0.75,
        fct_coster_kronig=1.0,
    )

    doublet = dublett_model.eval(params, x=x)

    assert np.all(np.isfinite(doublet))


def test_dublett_rejects_nonuniform_grid(dublett_model):
    """Materially nonuniform grids remain invalid for FFT convolution."""
    x = np.linspace(80.0, 68.0, 121)
    x[60:] -= 0.02
    params = dublett_model.make_params(
        amplitude=1.0,
        sigma=0.2,
        gamma=0.0,
        gaussian_sigma=0.67,
        center=71.25,
        soc=3.33,
        height_ratio=0.75,
        fct_coster_kronig=1.0,
    )

    with pytest.raises(ValueError, match="uniformly spaced"):
        dublett_model.eval(params, x=x)
