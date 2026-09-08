"""Physical coefficient bounds and positive-integrand Shirley option."""
import numpy as np
import pytest
from lmfit.model import save_model, load_model
from lmfit.models import GaussianModel
from lmfitxps.backgrounds import shirley, shirley_diagnostics
from lmfitxps.models import ShirleyBG


@pytest.mark.parametrize('monotonic', [False, True])
@pytest.mark.parametrize('k', [1., 2.])
def test_unreachable_endpoint_is_capped(monotonic, k):
    x = np.arange(4)*.1
    y = np.array([3., 2., -1., 0.])
    b = shirley(y, k, 0, x=x, monotonic=monotonic, alpha_max=.3)
    d = shirley_diagnostics(y, k, 0, x=x, monotonic=monotonic, alpha_max=.3)
    assert d['cap_active']
    assert d['alpha'] == pytest.approx(.3)
    assert d['q'] == pytest.approx(.03)
    residual = y-b
    if monotonic:
        residual = np.maximum(residual, 0)
        assert np.all(np.diff(b) <= 0)
        assert np.all(b >= 0)
    np.testing.assert_allclose(b, .03*np.r_[np.cumsum(residual[:-1][::-1])[::-1], 0], atol=1e-14)
    assert b[0] < k*y[0]


def test_signed_and_monotonic_are_distinct():
    y = [1., 3., -1., 0.]
    x = np.arange(4)
    assert shirley(y, 1, 0, x=x, alpha_max=.3)[-2] < 0
    b = shirley(y, 1, 0, x=x, monotonic=True, alpha_max=None)
    assert b[-2] == 0
    assert b[0] == pytest.approx(1)
    assert np.all(np.diff(b) <= 0)


@pytest.mark.parametrize('direction', [-1, 1])
def test_spacing_and_axis_direction(direction):
    y = [3., 2., -1., 0.]
    x = direction*np.arange(4)*.1
    b = shirley(y, 1, 0, x=x, alpha_max=.3)
    np.testing.assert_allclose(b, shirley(y, 1, 0, x=x*2, alpha_max=.15))


@pytest.mark.parametrize('kwargs', [{'alpha_max':.3}, {'x':[0, 1]}, {'x':[0, 1, 3]},
                                   {'x':[0, 1, 0]}, {'x':[0, 1, np.nan]},
                                   {'x':[0, 1, 2], 'alpha_max':0},
                                   {'x':[0, 1, 2], 'alpha_max':np.inf},
                                   {'x':[0, 1, 2], 'monotonic':'yes'}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        shirley([1, 2, 0], 1, 0, **kwargs)


def test_two_roots_within_cap_selects_first():
    y = np.array([.35, 1., 0.])  # roots t=.25, .4
    k = .75*.6/y[0]
    b = shirley(y, k, 0, x=np.arange(3), alpha_max=4.)
    np.testing.assert_allclose(b, [.45, .6, 0], atol=1e-12)
    assert not shirley_diagnostics(y, k, 0, x=np.arange(3), alpha_max=4.)['cap_active']


@pytest.mark.parametrize('monotonic', [False, True])
def test_fit_and_serialization(tmp_path, monotonic):
    x = np.linspace(10, 0, 101)
    peak = GaussianModel(prefix='p_')
    primary = peak.eval(x=x, amplitude=4, center=5, sigma=.6)
    y = primary + 2 + .01*np.r_[np.cumsum(primary[:-1][::-1])[::-1], 0]
    model = peak + ShirleyBG(prefix='b_', monotonic=monotonic)
    assert set(model.param_names) == {'p_amplitude', 'p_center', 'p_sigma', 'b_k', 'b_const'}
    params = model.make_params(p_amplitude=3, p_center=4.8, p_sigma=.7, b_k=1, b_const=2)
    params['b_const'].set(vary=False)
    params['b_k'].set(max=2)
    result = model.fit(y, params, x=x, y=y, max_nfev=400)
    assert result.success
    np.testing.assert_allclose(result.best_fit, y, atol=1e-5)
    path = tmp_path/'model.json'
    save_model(model, path)
    restored = load_model(path, funcdefs={'shirley':shirley})
    np.testing.assert_allclose(restored.eval(result.params, x=x, y=y), result.best_fit)
    assert restored.right.opts['monotonic'] == monotonic
    assert restored.right.opts['alpha_max'] == 'auto'


def test_diagnostics_resolve_prefix_and_expressions():
    model = ShirleyBG(prefix='bg_', monotonic=True, alpha_max=.3)
    params = model.make_params(k=1, const=0)
    params.add('endpoint_scale', value=2)
    params['bg_k'].set(expr='endpoint_scale')
    d = model.eval_diagnostics(params, x=np.arange(4)*.1, y=[3, 2, -1, 0])
    assert d['cap_active']
    assert d['requested_endpoint'] == 6
    assert d['alpha'] == pytest.approx(.3)


@pytest.mark.parametrize('monotonic', [False, True])
def test_cap_transition_is_continuous(monotonic):
    y = np.array([3., 2., -1., 0.])
    x = np.arange(4)*.1
    backgrounds = np.array([shirley(y, k, 0, x=x, monotonic=monotonic, alpha_max=.3)
                            for k in np.linspace(0, .2, 201)])
    # This simple branch reaches the cap then stays constant.
    assert np.max(np.abs(np.diff(backgrounds, axis=0))) < .004
    np.testing.assert_array_equal(backgrounds[-1], backgrounds[-2])


def test_rounded_uniform_export_is_accepted():
    x = np.round(np.linspace(0, 12, 294), 4)
    b = shirley(np.ones_like(x), 1, 0, x=x, alpha_max=.3)
    d = shirley_diagnostics(np.ones_like(x), 1, 0, x=x, alpha_max=.3)
    assert np.isfinite(b).all()
    assert d['q'] == pytest.approx(.3*12/293)


def test_bad_capped_root_cannot_escape(monkeypatch):
    from lmfitxps import backgrounds
    monkeypatch.setattr(backgrounds, 'brentq', lambda *args, **kwargs: .5)
    with pytest.raises(ValueError, match='self-consistency'):
        shirley([.35, 1, 0], .45/.35, 0, x=np.arange(3), alpha_max=4)
