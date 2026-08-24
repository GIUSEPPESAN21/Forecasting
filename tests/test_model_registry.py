"""F04 - despacho por clave exacta; ningun modelo se sustituye en silencio.

El defecto original: `_make_predictor` resolvia el metodo con
`if "arima" in nombre.lower()` y terminaba en `return train[-1]` (paseo
aleatorio) para cualquier nombre no reconocido — incluidos Promedio Simple,
Movil y Ponderado. El "MAPE optimizado" que el Modulo 3 reportaba para esos
metodos era el de un modelo completamente distinto.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.models import (
    MODEL_REGISTRY, ModelSpec, available_keys, forecast_with, get_spec,
)
from conftest import make_series

TODOS = sorted(MODEL_REGISTRY)


def test_clave_desconocida_levanta_error():
    with pytest.raises(KeyError) as exc:
        get_spec("Promedio Movil (k=3)")     # el nombre-etiqueta del codigo viejo
    assert "Disponibles" in str(exc.value)


@pytest.mark.parametrize("nombre", [
    "arima", "ARIMA(1,1,1)", "SARIMA(1,1,1)(1,1,1)[12]", "SES (alpha=0.1)",
    "Holt-Winters", "Regresion Lineal", "", "promedio",
])
def test_ningun_nombre_del_codigo_viejo_resuelve_por_subcadena(nombre):
    """Antes, 'Promedio Simple' caia al fallback naive sin avisar."""
    with pytest.raises(KeyError):
        get_spec(nombre)


def test_no_existe_fallback_silencioso_a_naive():
    """Ninguna clave valida puede producir el mismo resultado que naive por accidente."""
    y = make_series("tendencia", 48).to_numpy()
    naive = MODEL_REGISTRY["naive"].forecast(y, h=1, m=12)[0]
    distintos = 0
    for key in TODOS:
        if key == "naive":
            continue
        try:
            valor = MODEL_REGISTRY[key].forecast(y, h=1, m=12)[0]
        except Exception:
            continue
        distintos += int(abs(valor - naive) > 1e-9)
    assert distintos >= len(TODOS) - 3, (
        "demasiados modelos devuelven exactamente el valor naive: posible fallback"
    )


@pytest.mark.parametrize("key", TODOS)
def test_cada_modelo_expone_el_contrato_completo(key):
    spec = MODEL_REGISTRY[key]
    assert isinstance(spec, ModelSpec)
    assert spec.key == key
    assert spec.label and spec.family
    assert isinstance(spec.seasonal, bool)
    assert isinstance(spec.constant_level, bool)
    assert spec.min_obs(12) >= 1


@pytest.mark.parametrize("key", TODOS)
@pytest.mark.parametrize("h", [1, 6, 12])
def test_forecast_devuelve_exactamente_h_valores(key, h):
    spec = MODEL_REGISTRY[key]
    y = make_series("tendencia_estacional", 60).to_numpy()
    out = spec.forecast(y, h=h, m=12)
    assert out.shape == (h,)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("key", TODOS)
def test_la_misma_funcion_sirve_para_backtest_y_pronostico_final(key):
    """El original tenia 3 rutas distintas por metodo, que discrepaban entre si."""
    spec = MODEL_REGISTRY[key]
    y = make_series("tendencia_estacional", 60).to_numpy()
    un_paso = spec.forecast(y, h=1, m=12)[0]
    doce_pasos = spec.forecast(y, h=12, m=12)[0]
    assert un_paso == pytest.approx(doce_pasos), (
        "{}: el primer paso de un pronostico de 12 difiere del pronostico de 1".format(key)
    )


@pytest.mark.parametrize("key", TODOS)
def test_el_piso_de_no_negatividad_se_aplica_igual_en_todas_partes(key):
    """F21: la metrica reportada debe corresponder a lo que el usuario recibe."""
    spec = MODEL_REGISTRY[key]
    y = np.concatenate([np.linspace(1000, 10, 30), np.full(6, 5.0)])
    out = spec.forecast(y, h=12, m=12, clip_non_negative=True)
    assert (out >= 0).all(), "{} devolvio demanda negativa pese al piso".format(key)


def test_benchmarks_presentes_en_el_registro():
    assert "naive" in MODEL_REGISTRY and "seasonal_naive" in MODEL_REGISTRY


def test_los_nueve_metodos_del_manuscrito_siguen_disponibles():
    esperados = {
        "mean", "moving_average", "weighted_moving_average", "ses", "holt",
        "holt_winters", "linear_regression", "auto_arima", "auto_sarima",
    }
    faltantes = esperados - set(available_keys())
    assert not faltantes, "faltan metodos del manuscrito: {}".format(faltantes)


def test_forecast_with_es_equivalente_a_get_spec():
    y = make_series("plana", 36).to_numpy()
    a = forecast_with("ses", y, {"alpha": 0.3}, h=2, m=12)
    b = get_spec("ses").forecast(y, {"alpha": 0.3}, h=2, m=12)
    np.testing.assert_allclose(a, b)


def test_las_parrillas_estan_acotadas():
    """F13: Holt evaluaba 361 combinaciones sobre 18 observaciones."""
    for key in TODOS:
        spec = MODEL_REGISTRY[key]
        n_combos = len(spec.grid(48, 12))
        assert n_combos <= 40, "{} genera {} combinaciones".format(key, n_combos)


def test_arima_y_sarima_no_barren_parrilla():
    """Se seleccionan por AICc dentro de AutoARIMA, no por MAPE post-hoc."""
    for key in ("auto_arima", "auto_sarima"):
        if key in MODEL_REGISTRY:
            assert MODEL_REGISTRY[key].has_hyperparameters is False
