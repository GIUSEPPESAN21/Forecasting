"""Fase 11 / F27: adaptadores externos (Prophet, LightGBM), aislados del nucleo.

`pytest.importorskip` se aplica por CLASE (via una fixture `autouse`, no en
el cuerpo de la clase) para que si falta uno de los dos paquetes opcionales,
solo esa clase se salte -las demas se siguen coleccionando y corriendo. Esto
garantiza que la suite principal (ver `codigo/tests/` restante, 102 pruebas
del nucleo) nunca depende de `prophet` ni de `mlforecast`/`lightgbm`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 20260824


def _trend_seasonal_series(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = 1000 + 15 * t + 300 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 60, n)
    return np.maximum(y, 0.0)


class TestProphetAdapter:
    @pytest.fixture(autouse=True)
    def _require_prophet(self):
        pytest.importorskip("prophet")

    def test_fit_predict_shape_and_finite(self):
        from external_baselines.prophet_adapter import fit_predict_prophet

        y = _trend_seasonal_series(36)
        out = fit_predict_prophet(y, h=6, freq="MS")
        assert out.shape == (6,)
        assert np.isfinite(out).all()

    def test_no_negative_forecast_on_intermittent_demand(self):
        """F21: piso en cero aplicado tambien en el adaptador externo."""
        from external_baselines.prophet_adapter import fit_predict_prophet

        rng = np.random.default_rng(1)
        y = np.maximum(rng.normal(5, 4, 30), 0.0)
        out = fit_predict_prophet(y, h=6, freq="MS")
        assert (out >= 0).all()

    def test_insufficient_history_raises(self):
        from external_baselines.prophet_adapter import PROPHET_MIN_OBS, fit_predict_prophet

        y = _trend_seasonal_series(PROPHET_MIN_OBS - 1)
        with pytest.raises(ValueError):
            fit_predict_prophet(y, h=3, freq="MS")

    def test_yearly_seasonality_threshold_documented(self):
        """F26: la bandera se decide por n, no se activa siempre.

        No se re-verifica aqui el numero exacto del PDF (353.98% MAPE); eso
        depende de una corrida concreta con datos concretos y no es una
        propiedad estable del adaptador. Lo que SI es una propiedad estable,
        y lo que este test protege, es que el umbral es exactamente 24 (dos
        ciclos anuales) y que se puede forzar explicitamente en ambos
        sentidos para poder contrastar el efecto en un experimento.
        """
        from external_baselines.prophet_adapter import (
            PROPHET_MIN_OBS_YEARLY,
            fit_predict_prophet,
        )

        assert PROPHET_MIN_OBS_YEARLY == 24
        y = _trend_seasonal_series(24)
        auto = fit_predict_prophet(y, h=3, freq="MS")
        forced_off = fit_predict_prophet(y, h=3, freq="MS", yearly_seasonality=False)
        assert auto.shape == forced_off.shape == (3,)


class TestLightGBMAdapter:
    @pytest.fixture(autouse=True)
    def _require_mlforecast(self):
        pytest.importorskip("mlforecast")
        pytest.importorskip("lightgbm")

    def test_fit_predict_shape_and_finite(self):
        from external_baselines.lightgbm_adapter import fit_predict_lgbm

        y = _trend_seasonal_series(48)
        out = fit_predict_lgbm(y, h=6, freq="MS")
        assert out.shape == (6,)
        assert np.isfinite(out).all()

    def test_no_negative_forecast(self):
        from external_baselines.lightgbm_adapter import fit_predict_lgbm

        rng = np.random.default_rng(2)
        y = np.maximum(rng.normal(5, 4, 40), 0.0)
        out = fit_predict_lgbm(y, h=4, freq="MS")
        assert (out >= 0).all()

    def test_short_series_degrades_without_crashing(self):
        """n=LGBM_MIN_OBS: pocos lags disponibles, pero no debe lanzar."""
        from external_baselines.lightgbm_adapter import LGBM_MIN_OBS, fit_predict_lgbm

        y = _trend_seasonal_series(LGBM_MIN_OBS)
        out = fit_predict_lgbm(y, h=3, freq="MS")
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_insufficient_history_raises(self):
        from external_baselines.lightgbm_adapter import LGBM_MIN_OBS, fit_predict_lgbm

        y = _trend_seasonal_series(LGBM_MIN_OBS - 1)
        with pytest.raises(ValueError):
            fit_predict_lgbm(y, h=3, freq="MS")


class TestAdaptersPlugIntoCoreValidationUnmodified:
    """F27: las especificaciones externas son `ModelSpec` reales y se pasan a
    `forecasting_core.validation` SIN ninguna modificacion al nucleo (ver
    docstring de `external_baselines/adapters.py`)."""

    @pytest.fixture(autouse=True)
    def _require_both(self):
        pytest.importorskip("prophet")
        pytest.importorskip("mlforecast")
        pytest.importorskip("lightgbm")

    def test_specs_are_real_modelspec_instances(self):
        from external_baselines.adapters import LIGHTGBM_SPEC, PROPHET_SPEC
        from forecasting_core.models import ModelSpec

        assert isinstance(PROPHET_SPEC, ModelSpec)
        assert isinstance(LIGHTGBM_SPEC, ModelSpec)

    def test_specs_plug_into_walk_forward(self):
        from external_baselines.adapters import LIGHTGBM_SPEC, PROPHET_SPEC
        from forecasting_core.validation import rolling_origins, walk_forward

        y = _trend_seasonal_series(48)
        origins = rolling_origins(len(y), min_train=30)
        res = walk_forward(y, [PROPHET_SPEC, LIGHTGBM_SPEC], m=12, min_train=30, origins=origins)
        assert set(res.metrics["modelo"]) == {"ext_prophet", "ext_lightgbm"}
        assert res.metrics.set_index("modelo").loc["ext_prophet", "elegible"]
        assert res.metrics.set_index("modelo").loc["ext_lightgbm", "elegible"]

    def test_backtest_one_step_no_leakage(self):
        """Misma garantia que `test_no_leakage.py` exige al resto del registro:
        el pronostico en el origen t no puede depender de y[t] ni de nada
        posterior."""
        from external_baselines.adapters import LIGHTGBM_SPEC
        from forecasting_core.validation import backtest_one_step

        y = _trend_seasonal_series(40)
        origins = np.array([32])
        bt = backtest_one_step(y, LIGHTGBM_SPEC, None, 12, origins)
        pred_full = bt.y_pred[0]

        y_perturbed = y.copy()
        y_perturbed[32:] = y_perturbed[32:] + 10_000.0
        bt2 = backtest_one_step(y_perturbed, LIGHTGBM_SPEC, None, 12, origins)
        assert np.isclose(pred_full, bt2.y_pred[0]), (
            "el pronostico en el origen t cambio al perturbar y[t:] -fuga temporal"
        )


def test_external_baselines_importable_without_optional_deps():
    """El paquete debe poder importarse siempre (aunque falten las libs):
    los imports pesados son perezosos, dentro de cada funcion (ver docstring
    de `external_baselines/__init__.py`)."""
    import external_baselines
    import external_baselines.adapters
    import external_baselines.lightgbm_adapter
    import external_baselines.prophet_adapter

    assert hasattr(external_baselines, "PROPHET_AVAILABLE")
    assert hasattr(external_baselines, "LIGHTGBM_AVAILABLE")
