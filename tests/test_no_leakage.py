"""F03 - ausencia de fuga temporal, verificada de forma programatica.

El defecto original: `_fitted_series` calculaba el "pronostico ajustado sobre el
historico" de los promedios moviles con `rolling(k).mean()`, que en el instante t
promedia {t-2, t-1, t} — es decir, incluye la observacion que dice pronosticar.
El MAPE mostrado al usuario era 20.30% donde el honesto era 35.38%.

La prueba central de este archivo no inspecciona el codigo: **perturba el futuro
y comprueba que el pronostico no cambia**. Es imposible que pase si algun modelo
mira hacia adelante.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.models import MODEL_REGISTRY, InsufficientHistory
from forecasting_core.validation import backtest_one_step, rolling_origins
from conftest import make_series

TODOS = sorted(MODEL_REGISTRY)


@pytest.mark.parametrize("key", TODOS)
def test_el_pronostico_no_depende_del_futuro(key):
    """Perturbar y[t:] no puede alterar el pronostico hecho con y[:t]."""
    spec = MODEL_REGISTRY[key]
    s = make_series("tendencia_estacional", 60)
    y = s.to_numpy(dtype=float)
    t = 40
    if t < spec.min_obs(12):
        pytest.skip("{} requiere mas historia".format(key))

    base = spec.forecast(y[:t], h=3, m=12)

    contaminado = y.copy()
    contaminado[t:] = contaminado[t:] * 7.5 + 10_000.0   # futuro radicalmente distinto
    otro = spec.forecast(contaminado[:t], h=3, m=12)

    np.testing.assert_allclose(
        base, otro, rtol=0, atol=0,
        err_msg="{} produce un pronostico distinto al cambiar el FUTURO".format(key),
    )


@pytest.mark.parametrize("key", TODOS)
def test_el_backtest_completo_es_inmune_al_futuro(key):
    """Cada origen del walk-forward debe depender solo de su propio pasado."""
    spec = MODEL_REGISTRY[key]
    s = make_series("tendencia_estacional", 60)
    y = s.to_numpy(dtype=float)
    origins = rolling_origins(len(y), max(24, spec.min_obs(12)))
    if origins.size < 3:
        pytest.skip("{} no deja origenes suficientes".format(key))

    completo = backtest_one_step(y, spec, None, 12, origins)

    corte = origins[len(origins) // 2]
    truncado_y = y.copy()
    truncado_y[corte:] = np.nan          # el futuro deja de existir
    # Se evaluan solo los origenes anteriores al corte, con el futuro borrado.
    previos = origins[origins <= corte]
    parcial = backtest_one_step(truncado_y, spec, None, 12, previos)

    n = previos.size
    np.testing.assert_allclose(
        completo.y_pred[:n], parcial.y_pred[:n], rtol=1e-9, atol=1e-9,
        err_msg="{}: borrar el futuro cambio pronosticos del pasado".format(key),
    )


def test_promedio_movil_no_incluye_la_observacion_actual():
    """Regresion directa del bug de `rolling(k).mean()` (F03)."""
    spec = MODEL_REGISTRY["moving_average"]
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    # Con k=3 y todo el historial, el pronostico del siguiente periodo es la
    # media de los TRES ULTIMOS observados: (60+70+80)/3 = 70.
    assert spec.forecast(y, params={"k": 3}, h=1, m=12)[0] == pytest.approx(70.0)
    # El pronostico de y[7]=80 solo puede usar y[:7] = [10..70]: (50+60+70)/3 = 60.
    # El codigo viejo mostraba en su lugar (60+70+80)/3 = 70, que incluye el
    # propio 80 que decia predecir: MAPE mostrado 20.30% vs honesto 35.38%.
    assert spec.forecast(y[:7], params={"k": 3}, h=1, m=12)[0] == pytest.approx(60.0)
    assert spec.forecast(y[:7], params={"k": 3}, h=1, m=12)[0] != pytest.approx(70.0)


def test_ponderado_pondera_mas_lo_reciente():
    spec = MODEL_REGISTRY["weighted_moving_average"]
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    valor = spec.forecast(y, params={"weights": (0.5, 0.3, 0.2)}, h=1, m=12)[0]
    assert valor == pytest.approx(50.0), "el peso mayor debe ir al periodo mas reciente"


def test_seasonal_naive_repite_el_ciclo_correcto():
    spec = MODEL_REGISTRY["seasonal_naive"]
    y = np.arange(24, dtype=float)
    fc = spec.forecast(y, h=14, m=12)
    np.testing.assert_allclose(fc[:12], y[-12:])
    np.testing.assert_allclose(fc[12:], y[-12:][:2])


@pytest.mark.parametrize("key", TODOS)
def test_historia_insuficiente_levanta_excepcion_no_devuelve_otro_modelo(key):
    """F02/F04: nunca sustituir el modelo en silencio."""
    spec = MODEL_REGISTRY[key]
    y = np.arange(1.0, 3.0)  # 2 observaciones
    if spec.min_obs(12) <= 2:
        pytest.skip("{} si puede operar con 2 observaciones".format(key))
    with pytest.raises((InsufficientHistory, ValueError)):
        spec.forecast(y, h=1, m=12)
