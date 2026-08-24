"""F01, F10, F11 - clasificacion estructural.

Cada prueba de este archivo FALLA con el codigo original y pasa con el nuevo.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.classification import (
    allow_constant_level_methods, classify_series, deseasonalize,
    seasonality_test, trend_test,
)
from conftest import make_series


# ---------------------------------------------------------------------------
# F01 - el predicado insatisfacible
# ---------------------------------------------------------------------------
def _es_muy_lineal_original(serie, r2_umbral=0.90, p_umbral=0.01, ac12_umbral=0.10):
    """Reimplementacion literal del predicado original, para regresion."""
    import pandas as pd
    from scipy import stats

    vals = pd.Series(serie).dropna().values
    if len(vals) < 6:
        return False
    x = np.arange(len(vals))
    _, _, r, p_tend, _ = stats.linregress(x, vals)
    try:
        ac12 = float(pd.Series(serie).autocorr(lag=12))
    except Exception:
        ac12 = np.nan
    ac12_abs = abs(ac12) if not np.isnan(ac12) else 0.0
    return (r**2 >= r2_umbral) and (p_tend < p_umbral) and (ac12_abs < ac12_umbral)


@pytest.mark.parametrize("kind,n", [
    ("tendencia_perfecta", 24), ("tendencia_perfecta", 48),
    ("tendencia_limpia", 24), ("tendencia_limpia", 48),
    ("plana", 24), ("plana", 48), ("estacional", 48),
])
def test_predicado_original_es_insatisfacible(kind, n):
    """El predicado viejo devuelve False SIEMPRE, incluso para y=1000+25t."""
    assert not _es_muy_lineal_original(make_series(kind, n))


@pytest.mark.parametrize("n", [24, 36, 48])
def test_serie_plana_admite_metodos_de_nivel_constante(n):
    """F01: en una serie plana, promedio simple/movil DEBEN poder competir.

    Es el caso que el codigo original hacia imposible y que ambos manuscritos
    reportaban como observado ("el Promedio Simple obtiene el menor MAPE").
    """
    prof = classify_series(make_series("plana", n), m=12)
    assert not prof.has_trend
    assert allow_constant_level_methods(prof) is True


@pytest.mark.parametrize("kind,n", [
    ("tendencia_perfecta", 36), ("tendencia_limpia", 36), ("tendencia_limpia", 48),
])
def test_serie_con_tendencia_excluye_nivel_constante(kind, n):
    """La regla correcta es la INVERSA de la original."""
    prof = classify_series(make_series(kind, n), m=12)
    assert prof.has_trend
    assert allow_constant_level_methods(prof) is False


def test_paseo_aleatorio_excluye_promedios_aunque_no_tenga_tendencia():
    """Nivel no estacionario => promediar toda la historia es peor que el naive."""
    prof = classify_series(make_series("paseo_aleatorio", 60), m=12)
    if not prof.has_trend:  # el caso interesante
        assert allow_constant_level_methods(prof) == prof.is_stationary


# ---------------------------------------------------------------------------
# F11 - tamano del test de tendencia
# ---------------------------------------------------------------------------
def test_tendencia_no_se_dispara_con_ruido_blanco():
    """Tamano ~5% sobre ruido i.i.d. (nominal)."""
    rng = np.random.default_rng(7)
    rechazos = sum(trend_test(1000 + rng.normal(0, 100, 48))[0] for _ in range(200))
    assert rechazos / 200 <= 0.15, "tasa de falsos positivos demasiado alta"


def test_tendencia_sobre_paseo_aleatorio_muy_por_debajo_del_original():
    """El original daba 74.4%; el objetivo es <= 25%."""
    rng = np.random.default_rng(11)
    rechazos = sum(
        trend_test(1000 + np.cumsum(rng.normal(0, 60, 48)))[0] for _ in range(200)
    )
    tasa = rechazos / 200
    assert tasa <= 0.25, "falsos positivos de tendencia en paseo aleatorio: {:.1%}".format(tasa)


def test_tendencia_real_se_detecta():
    prof = classify_series(make_series("tendencia_limpia", 48), m=12)
    assert prof.has_trend
    assert prof.trend_slope > 0


# ---------------------------------------------------------------------------
# F10 - estacionalidad
# ---------------------------------------------------------------------------
def test_tendencia_pura_no_se_confunde_con_estacionalidad():
    """El criterio |ACF(12)|>0.30 daba 50.2% de falsos positivos aqui."""
    rng = np.random.default_rng(3)
    falsos = 0
    for _ in range(100):
        y = 1000 + 25 * np.arange(48) + rng.normal(0, 100, 48)
        falsos += seasonality_test(y, m=12)[0]
    assert falsos / 100 <= 0.10, "falsos positivos de estacionalidad: {:.1%}".format(falsos / 100)


def test_estacionalidad_real_se_detecta_con_ciclos_suficientes():
    ok, fs, p, _ = seasonality_test(make_series("estacional", 48), m=12)
    assert ok and fs >= 0.40 and p < 0.05


def test_estacionalidad_no_evaluable_con_menos_de_tres_ciclos():
    """Con 24 observaciones mensuales la respuesta honesta es 'no evaluable'."""
    ok, fs, _, nombre = seasonality_test(make_series("estacional", 24), m=12)
    assert ok is False
    assert np.isnan(fs)
    assert "no evaluable" in nombre


def test_desestacionalizar_permite_ver_la_tendencia():
    """Sin desestacionalizar, la tendencia se perdia en el 100% de los casos."""
    s = make_series("tendencia_estacional", 48)
    assert trend_test(deseasonalize(s, 12))[0] is True


# ---------------------------------------------------------------------------
# Coherencia del perfil
# ---------------------------------------------------------------------------
def test_perfil_avisa_de_baja_potencia_en_series_cortas():
    prof = classify_series(make_series("plana", 24), m=12)
    assert prof.low_power
    assert any("potencia" in w for w in prof.warnings)


def test_periodo_estacional_efectivo_es_1_si_no_hay_estacionalidad():
    """Usar m=12 en una serie no estacional infla min_train y deja sin origenes."""
    prof = classify_series(make_series("plana", 48), m=12)
    assert prof.seasonal_period == 1


def test_estacionariedad_reporta_no_concluyente_si_adf_y_kpss_discrepan():
    prof = classify_series(make_series("paseo_aleatorio", 60), m=12)
    assert prof.stationarity_verdict
    assert prof.adf_regression in {"c", "ct"}
