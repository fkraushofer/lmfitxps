import inspect

import numpy as np
from lmfit import Model
from lmfit.model import load_model, save_model

from lmfitxps import lineshapes, models


def test_legacy_public_names_are_exact_aliases():
    assert lineshapes.singlett is lineshapes.singlet
    assert lineshapes.dublett is lineshapes.doublet
    assert lineshapes.dublett_components is lineshapes.doublet_components
    assert lineshapes._dublett_oversampling is lineshapes._doublet_oversampling
    assert (
        models.ConvGaussianDoniachSinglett
        is models.ConvGaussianDoniachSinglet
    )
    assert (
        models.ConvGaussianDoniachDublett
        is models.ConvGaussianDoniachDoublet
    )
    assert models.dublett_ratio_diagnostics is models.doublet_ratio_diagnostics
    assert models.dublett_ratio_report is models.doublet_ratio_report
    assert (
        models.ConvGaussianDoniachDoublet.eval_dublett_components
        is models.ConvGaussianDoniachDoublet.eval_doublet_components
    )


def test_legacy_aliases_preserve_signatures_and_numerical_results():
    assert inspect.signature(lineshapes.singlett) == inspect.signature(
        lineshapes.singlet
    )
    assert inspect.signature(lineshapes.dublett) == inspect.signature(
        lineshapes.doublet
    )
    x = np.linspace(80.0, 68.0, 121)
    singlet_args = (1.0, 0.2, 0.02, 0.3, 72.0)
    doublet_args = (*singlet_args, 3.3, 0.75, 1.0)

    np.testing.assert_array_equal(
        lineshapes.singlett(x, *singlet_args),
        lineshapes.singlet(x, *singlet_args),
    )
    np.testing.assert_array_equal(
        lineshapes.dublett(x, *doublet_args),
        lineshapes.doublet(x, *doublet_args),
    )


def test_correct_model_names_keep_parameter_and_component_conventions():
    singlet = models.ConvGaussianDoniachSinglet(prefix="singlett_")
    doublet = models.ConvGaussianDoniachDoublet(prefix="dublett_")

    assert singlet.func.__name__ == "singlet"
    assert doublet.func.__name__ == "oversampled_doublet"
    assert singlet.param_names[0].startswith("singlett_")
    assert doublet.param_names[0].startswith("dublett_")
    assert set(doublet.param_names) >= {
        "dublett_amplitude",
        "dublett_center",
        "dublett_soc",
        "dublett_height_ratio",
    }


def test_historical_funcdefs_keys_restore_models():
    """lmfit files saved with the 4.x callable keys remain loadable."""
    x = np.linspace(80.0, 68.0, 121)
    args = dict(
        amplitude=1.0,
        sigma=0.2,
        gamma=0.02,
        gaussian_sigma=0.3,
        center=72.0,
    )
    current = Model(lineshapes.singlet)
    serialized = current.dumps()
    historical = serialized.replace(
        '"funcname": "singlet"', '"funcname": "singlett"'
    )
    assert historical != serialized
    restored = Model(lineshapes.singlet).loads(
        historical, funcdefs={"singlett": lineshapes.singlett}
    )

    np.testing.assert_array_equal(
        restored.eval(x=x, **args), current.eval(x=x, **args)
    )


def test_correct_models_round_trip_through_lmfit_serialization(tmp_path):
    x = np.linspace(80.0, 68.0, 121)
    singlet = models.ConvGaussianDoniachSinglet(prefix="s_")
    doublet = models.ConvGaussianDoniachDoublet(
        prefix="d_", max_oversampling=12
    )
    cases = (
        (
            singlet,
            singlet.make_params(
                amplitude=1.0,
                sigma=0.2,
                gamma=0.02,
                gaussian_sigma=0.3,
                center=72.0,
            ),
        ),
        (
            doublet,
            doublet.make_params(
                amplitude=1.0,
                sigma=0.2,
                gamma=0.02,
                gaussian_sigma=0.3,
                center=72.0,
                soc=3.3,
                height_ratio=0.75,
                fct_coster_kronig=1.0,
            ),
        ),
    )

    for index, (model, params) in enumerate(cases):
        path = tmp_path / f"model-{index}.json"
        save_model(model, path)
        restored = load_model(path)
        np.testing.assert_array_equal(
            restored.eval(params, x=x), model.eval(params, x=x)
        )


def test_new_and_legacy_doublet_diagnostics_match():
    x = np.linspace(80.0, 68.0, 121)
    model = models.ConvGaussianDoniachDoublet(prefix="peak_")
    params = model.make_params(
        amplitude=1.0,
        sigma=0.2,
        gamma=0.02,
        gaussian_sigma=0.3,
        center=72.0,
        soc=3.3,
        height_ratio=0.75,
        fct_coster_kronig=1.0,
    )

    new = model.eval_doublet_components(params, x)
    legacy = model.eval_dublett_components(params, x)
    np.testing.assert_array_equal(new[0], legacy[0])
    np.testing.assert_array_equal(new[1], legacy[1])
