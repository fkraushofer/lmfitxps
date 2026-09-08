"""Automatic 20% slope fallback and its user-visible warning."""
import warnings
import numpy as np
import pytest
from lmfitxps import backgrounds
from lmfitxps.models import ShirleyBG


def test_analytic_single_interval_limit_and_warning():
    # k(q)=q/(1+q); logarithmic sensitivity peaks at q=1.
    # q/(1+q)^2 = 0.2/4 has the descending-side root 9+sqrt(80).
    with pytest.warns(backgrounds.ShirleyBackgroundWarning, match='lower starting k'):
        b = backgrounds.shirley([1., 0.], 1, 0)
    d = backgrounds.shirley_diagnostics([1., 0.], 1, 0)
    expected = 9+np.sqrt(80)
    assert d['q_limit'] == pytest.approx(expected, rel=1e-7)
    assert d['q'] == d['q_limit']
    assert b[0] == pytest.approx(expected/(1+expected))
    assert d['cap_method'] == 'slope_20_percent'
    assert d['alpha_limit'] is None


@pytest.mark.parametrize('monotonic', [False, True])
def test_cap_independent_of_requested_k(monotonic):
    y = [3., 2., -.5, 0.]
    ds = [backgrounds.shirley_diagnostics(y, k, 0, monotonic=monotonic)
          for k in [.3, 1, 2, 100]]
    assert len({d['q_limit'] for d in ds}) == 1
    assert not ds[0]['cap_active']
    assert all(d['cap_active'] for d in ds[1:])
    assert len({d['q'] for d in ds[1:]}) == 1


@pytest.mark.parametrize('direction', [-1, 1])
def test_auto_spacing_conversion_and_intensity_invariance(direction):
    y = np.array([3., 2., -.5, 0.])
    x = direction*np.arange(4)*.1
    a = backgrounds.shirley_diagnostics(y, 1, 0, x=x)
    b = backgrounds.shirley_diagnostics(1e5*y+400, 1, 400, x=x*2)
    assert a['q_limit'] == pytest.approx(b['q_limit'])
    assert a['alpha_limit'] == pytest.approx(2*b['alpha_limit'])


def test_explicit_cap_and_uncapped_escape_hatch():
    a = backgrounds.shirley_diagnostics([1., 0.], 1, 0, x=[.1, 0], alpha_max=.3)
    assert a['q_limit'] == pytest.approx(.03)
    assert a['cap_method'] == 'explicit'
    with pytest.raises(ValueError, match='no finite'):
        backgrounds.shirley([1., 0.], 1, 0, alpha_max=None)


def test_clean_solution_is_not_changed_or_warned():
    x = np.linspace(10, 0, 101)
    p = np.exp(-.5*((x-5)/.6)**2)
    p[[0,-1]]=0
    b = 2+.01*np.r_[np.cumsum(p[:-1][::-1])[::-1],0]
    y = p+b
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = backgrounds.shirley(y, 1, 2, x=x)
    assert not caught
    np.testing.assert_allclose(actual, b, atol=1e-10)


def test_cached_limit_uses_data_and_const_but_not_k():
    backgrounds._shirley_auto_qmax.cache_clear()
    for k in [.1,.2,.3]:
        backgrounds.shirley_diagnostics([3,2,0], k, 0)
    assert backgrounds._shirley_auto_qmax.cache_info().misses == 1
    backgrounds.shirley_diagnostics([3,2,0], .2, -.5)
    backgrounds.shirley_diagnostics([3,2.1,0], .2, -.5)
    assert backgrounds._shirley_auto_qmax.cache_info().misses == 3


def test_default_warning_filter_deduplicates_and_diagnostics_are_quiet():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default', backgrounds.ShirleyBackgroundWarning)
        for k in [1.,2.,3.]:
            backgrounds.shirley([1,0], k, 0)
        backgrounds.shirley_diagnostics([1,0], 2, 0)
    assert len(caught) == 1


def test_fallback_does_not_mutate_lmfit_parameters():
    model = ShirleyBG(prefix='bg_')
    p = model.make_params(k=1.2, const=0)
    p['bg_k'].set(min=0,max=2)
    before = (p['bg_k'].value, p['bg_k'].min, p['bg_k'].max)
    with pytest.warns(backgrounds.ShirleyBackgroundWarning):
        model.eval(p, y=[1.,0.], x=[.1,0.])
    assert before == (p['bg_k'].value, p['bg_k'].min, p['bg_k'].max)


@pytest.mark.parametrize('invalid', ['bad', -1, float('nan')])
def test_invalid_limit(invalid):
    with pytest.raises(ValueError, match='alpha_max'):
        backgrounds.shirley([1,0], 1, 0, alpha_max=invalid)
