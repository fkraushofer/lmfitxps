"""Deterministic checks of the active Shirley equation and fit integration."""

import numpy as np
import pytest
from lmfit.model import load_model, save_model
from lmfit.models import GaussianModel

from lmfitxps.backgrounds import shirley as public_shirley
from functools import partial
shirley = partial(public_shirley, alpha_max=None)
from lmfitxps.models import ShirleyBG
from lmfitxps import backgrounds


def _known_spectrum(q=0.15):
    x = np.linspace(0, 10, 101)
    primary = np.exp(-0.5*((x-3)/0.5)**2)
    primary += 0.7*np.exp(-0.5*((x-6)/0.7)**2)
    primary[[0, -1]] = 0
    background = 2 + q*np.r_[np.cumsum(primary[:-1][::-1])[::-1], 0]
    return primary + background, background


def _assert_equation(y, background, k, const):
    residual = y-background
    cumulative = np.r_[np.cumsum(residual[:-1][::-1])[::-1], 0]
    expected = const + k*(y[0]-const)*cumulative/cumulative[0]
    np.testing.assert_allclose(background, expected, rtol=1e-9, atol=1e-10)
    assert background[0] == pytest.approx(const + k*(y[0]-const))
    assert background[-1] == const


def _legacy_iteration(y, k, const):
    """Old algorithm, used only to reproduce its convergence failure."""
    step = k*(y[0]-const)
    background = np.linspace(const+step, const, len(y))
    for _ in range(100):
        residual = y-background
        cumulative = np.r_[np.cumsum(residual[:-1][::-1])[::-1], 0]
        new = const + step*cumulative/cumulative[0]
        if np.allclose(new, background, rtol=1e-8, atol=1e-10):
            return new
        background = new
    return background


@pytest.mark.parametrize('q', [0.005, 0.03, 0.15])
def test_recovers_known_background(q):
    y, expected = _known_spectrum(q)
    actual = shirley(y, 1, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    _assert_equation(y, actual, 1, 2)


def test_solves_case_where_fixed_point_iteration_diverges():
    y, expected = _known_spectrum()
    old = _legacy_iteration(y, 1, 2)
    assert np.max(np.abs(old-expected)) > 0.1
    np.testing.assert_allclose(shirley(y, 1, 2), expected, atol=1e-10)


def test_agrees_with_converged_iteration():
    y, _ = _known_spectrum(0.005)
    np.testing.assert_allclose(shirley(y, 1, 2), _legacy_iteration(y, 1, 2),
                               rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('first,last', [(.2, .7), (.25, .5), (.4, .40001)])
def test_selects_smallest_positive_q_of_two_roots(first, last):
    # Endpoint polynomial -(t-first)*(t-last); q=(1-t)/t.
    # The closely spaced pair would be missed by a coarse sampling grid.
    y = np.array([1-first-last, 1.0, 0.0])
    target = (1-first)*(1-last)
    k = target/y[0]
    background = shirley(y, k, 0)
    np.testing.assert_allclose(background, [target, 1-last, 0], atol=1e-10)
    _assert_equation(y, background, k, 0)


def test_negative_noise_is_not_clipped():
    y = np.array([1., 3., -1., 0.])
    actual = shirley(y, 1, 0)
    assert actual[-2] < 0
    _assert_equation(y, actual, 1, 0)


def test_multiple_root_is_rejected():
    # -(t-.25)**2: a tangent root is not a reliably resolved fit branch.
    with pytest.raises(ValueError, match='multiple|ill-conditioned'):
        shirley([.5, 1, 0], 1.125, 0)


def test_incorrect_root_cannot_silently_escape(monkeypatch):
    monkeypatch.setattr(backgrounds, 'brentq', lambda *args, **kwargs: .5)
    y, _ = _known_spectrum()
    with pytest.raises(ValueError, match='self-consistency'):
        shirley(y, 1, 2)


@pytest.mark.parametrize('scale,offset', [(1e-9, 0), (1e9, 0), (1, 1e6), (-1, 0)])
def test_intensity_scaling_and_offset(scale, offset):
    y, expected = _known_spectrum()
    actual = shirley(scale*y+offset, 1, scale*2+offset)
    np.testing.assert_allclose((actual-offset)/scale, expected, atol=1e-8)


@pytest.mark.parametrize('y,k,const', [([10, 8, 6, 4], 1, 2),
                                      ([1, 1, 1], 2, 0), ([1, 0], 1, 0)])
def test_no_finite_solution_raises(y, k, const):
    with pytest.raises(ValueError, match='no finite positive-q solution'):
        shirley(y, k, const)


@pytest.mark.parametrize('y,k,const', [([[1, 2]], 1, 0), ([1, np.nan], 1, 0),
                                      ([1, 2], -1, 0), ([1, 2], np.inf, 0),
                                      ([1, 2], 1, np.nan)])
def test_invalid_inputs_raise(y, k, const):
    with pytest.raises(ValueError):
        shirley(y, k, const)


@pytest.mark.parametrize('y,k,const,expected', [([], 1, 2, []), ([5], 1, 2, [2]),
                                              ([2, 4, 2], 1, 2, [2, 2, 2]),
                                              ([3, 4, 2], 0, 2, [2, 2, 2])])
def test_constant_and_short_inputs(y, k, const, expected):
    np.testing.assert_array_equal(shirley(y, k, const), expected)


def test_parameter_sweep_is_continuous_and_order_independent():
    y, _ = _known_spectrum()
    ks = np.linspace(0.7, 1., 101)
    forward = np.array([shirley(y, k, 2) for k in ks])
    backward = np.array([shirley(y, k, 2) for k in ks[::-1]])[::-1]
    np.testing.assert_array_equal(forward, backward)
    finer = np.array([shirley(y, k, 2) for k in np.linspace(.7, 1, 201)])
    # Halving the parameter step shrinks changes, rather than retaining a jump.
    assert np.max(np.abs(np.diff(finer, axis=0))) < .7*np.max(
        np.abs(np.diff(forward, axis=0)))


@pytest.mark.parametrize('direction', [-1, 1])
def test_composite_fit_and_model_serialization(tmp_path, direction):
    # Descending binding energy and ascending kinetic energy, with the
    # same intensity order and corresponding mirrored peak coordinates.
    x = direction*np.linspace(0, 10, 101)
    peak_model = GaussianModel(prefix='peak_')
    peak = peak_model.eval(x=x, amplitude=3, center=direction*5, sigma=.6)
    background = 2 + .08*np.r_[np.cumsum(peak[:-1][::-1])[::-1], 0]
    y = peak+background
    model = peak_model + ShirleyBG(prefix='bg_', alpha_max=None)
    params = model.make_params(peak_amplitude=2.5, peak_center=direction*4.8,
                               peak_sigma=.7, bg_k=.9, bg_const=2)
    params['bg_const'].set(vary=False)
    params['bg_k'].set(min=.5, max=1)
    result = model.fit(y, params, x=x, y=y, max_nfev=400)
    assert result.success
    assert result.nfev < 400
    np.testing.assert_allclose(result.best_fit, y, atol=1e-5)
    assert result.params['peak_center'].value == pytest.approx(direction*5, abs=1e-5)
    assert result.params['peak_amplitude'].value == pytest.approx(3, abs=1e-4)
    path = tmp_path / 'model.json'
    save_model(model, path)
    restored = load_model(path, funcdefs={'shirley': public_shirley})
    assert set(restored.param_names) == set(model.param_names)
    np.testing.assert_allclose(restored.eval(result.params, x=x, y=y), result.best_fit)
