"""The same spectrum uses ascending KE or descending BE, with y unchanged."""

import numpy as np
import pytest

from src.lmfitxps import backgrounds, models


ENERGY_SUM = 1486.6


def peak_model(kind):
    classes = {
        'singlet': models.ConvGaussianDoniachSinglett,
        'doublet': models.ConvGaussianDoniachDublett,
        'fermi': models.FermiEdgeModel,
    }
    model = classes[kind](prefix='p_')
    values = dict(amplitude=20., center=95., sigma=0.3)
    if kind == 'fermi':
        values['kt'] = 0.1
    else:
        values.update(gamma=0.12, gaussian_sigma=0.25)
    if kind == 'doublet':
        values.update(soc=3.2, height_ratio=0.65, fct_coster_kronig=1.2)
    return model, values


def background_model(kind):
    if kind == 'tougaard':
        model = models.TougaardBG(prefix='b_', independent_vars=['x', 'y'])
        # Each evaluation must compute its own integral: a shared stale cache
        # could make an incorrect BE calculation appear identical to KE.
        model.func = backgrounds.tougaard_closure()
        return model, dict(B=30., C=20., C_d=1., D=10., extend=5.)
    if kind == 'shirley':
        return models.ShirleyBG(prefix='b_'), dict(k=0.3, const=1.)
    return models.SlopeBG(prefix='b_', independent_vars=['y']), dict(k=0.0001)


@pytest.mark.parametrize('size', [200, 201])
@pytest.mark.parametrize('peak', ['singlet', 'doublet', 'fermi'])
@pytest.mark.parametrize('background', [None, 'tougaard', 'shirley', 'slope'])
def test_equivalent_energy_fits(peak, background, size):
    """Recover shared shape parameters and reflected centers from one dataset."""
    kinetic = np.linspace(85., 105., size)
    binding = ENERGY_SUM - kinetic
    # A fixed synthetic measured spectrum supplies the background integrals.
    measured = 2. + 3. * (kinetic[-1] - kinetic) / np.ptp(kinetic)
    measured += 20. * np.exp(-((kinetic - 95.) / 1.5)**2)
    fitted = []
    reference = None
    for x, is_binding in [(kinetic, False), (binding, True)]:
        model, values = peak_model(peak)
        if is_binding:
            values['center'] = ENERGY_SUM - values['center']
        params = model.make_params(**values)
        if background:
            bg, bg_values = background_model(background)
            model = model + bg
            params.update(bg.make_params(**bg_values))
        inputs = dict(x=x)
        if background:
            inputs['y'] = measured
        truth = model.eval(params, **inputs)
        if reference is None:
            reference = truth
        else:
            np.testing.assert_allclose(truth, reference, rtol=1e-10, atol=1e-10)

        # Vary identifiable parameters, keeping background shape parameters
        # fixed as in a broad-region calibration / narrow-region fit workflow.
        varying = ['p_amplitude', 'p_center', 'p_sigma']
        if peak != 'fermi':
            varying += ['p_gamma']
        if background:
            varying += ['b_B' if background == 'tougaard' else 'b_k']
        for name, param in params.items():
            if param.expr is None:
                param.vary = name in varying
        for name in varying:
            params[name].value *= 0.85 if name != 'p_center' else 1.
        params['p_center'].value += -0.15 if is_binding else 0.15
        if peak != 'fermi':
            params['p_gamma'].set(min=0., max=0.5)
        result = model.fit(reference, params, **inputs)
        assert result.success
        np.testing.assert_allclose(result.best_fit, reference, rtol=1e-6, atol=1e-7)
        expected_center = ENERGY_SUM - 95. if is_binding else 95.
        assert result.params['p_center'].value == pytest.approx(expected_center, abs=1e-6)
        if peak != 'fermi':
            assert result.params['p_gamma'].value == pytest.approx(0.12, abs=1e-6)
        fitted.append(result)
    for name in varying:
        left, right = [result.params[name].value for result in fitted]
        if name == 'p_center':
            right = ENERGY_SUM - right
        assert left == pytest.approx(right, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize('peak', ['singlet', 'doublet'])
def test_positive_gamma_has_high_binding_energy_tail(peak):
    """Check physical direction independently of the KE/BE comparison."""
    x = np.linspace(115., 75., 2001)
    model, values = peak_model(peak)
    params = model.make_params(**values)
    if peak == 'doublet':
        profiles = model.eval_dublett_components(params, x=x)
        # Both components must be reflected about their own centers.
        profiles = zip(profiles, [95., 98.2])
    else:
        profiles = [(model.eval(params, x=x), 95.)]
    for profile, center in profiles:
        low = profile[np.argmin(abs(x - (center - 2.)))]
        high = profile[np.argmin(abs(x - (center + 2.)))]
        assert high > 2 * low > 0


@pytest.mark.parametrize('extend', [0., 5.])
def test_tougaard_fresh_evaluation_energy_equivalence(extend):
    x = np.linspace(85., 105., 101)
    y = 2. + np.exp(-((x - 95.) / 2.)**2)
    kw = dict(y=y, B=30., C=20., C_d=1., D=10., extend=extend)
    kinetic = backgrounds.tougaard_closure()(x=x, **kw)
    binding = backgrounds.tougaard_closure()(x=ENERGY_SUM-x, **kw)
    np.testing.assert_allclose(binding, kinetic, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('changed', ['x', 'C', 'C_d', 'D'])
def test_tougaard_cache_tracks_integral_inputs(changed):
    x = np.linspace(85., 105., 101)
    kw = dict(x=x, y=2.+np.exp(-((x-95.)/2.)**2),
              B=30., C=20., C_d=1., D=10., extend=5.)
    cached = backgrounds.tougaard_closure()
    original = cached(**kw)
    kw[changed] = kw[changed] * 1.5
    expected = backgrounds.tougaard_closure()(**kw)
    assert not np.allclose(original, expected)
    np.testing.assert_allclose(cached(**kw), expected)


@pytest.mark.parametrize('method', ['shirley', 'tougaard'])
def test_static_background_energy_equivalence(method):
    x = np.linspace(85., 105., 81)
    y = 2.+0.05*(105.-x)+10.*np.exp(-((x-95.)/2.)**2)
    if method == 'shirley':
        def calculate(energy):
            return backgrounds.shirley_calculate(energy, y, maxit=100, tol=1e-10)
    else:
        def calculate(energy):
            return backgrounds.tougaard_calculate(
                energy, y, tb=30., tc=20., tcd=1., td=10., maxit=50)[0]
    np.testing.assert_allclose(calculate(x), calculate(ENERGY_SUM-x),
                               rtol=1e-10, atol=1e-10)
