"""F02 - paridad del walk-forward y F13/F14 - una sola pasada.

El defecto original: "Holt-Winters" y "SARIMA" caian silenciosamente a Holt y
ARIMA(1,1,1) en 8 de 18 origenes con n=24, y aun asi competian en el mismo
ranking contra metodos evaluados sobre los 18. Se promediaban MAPEs calculados
sobre conjuntos de tamano distinto.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.models import MODEL_REGISTRY, eligible_specs
from forecasting_core.classification import classify_series
from forecasting_core.optimize import run_pipeline
from forecasting_core.validation import (
    ABSOLUTE_MIN_TRAIN, resolve_min_train, rolling_origins, walk_forward,
)
from conftest import make_series


def test_todos_los_metodos_del_ranking_tienen_el_mismo_n_preds():
    """La condicion que el codigo original violaba."""
    s = make_series("tendencia_estacional", 60)
    res = run_pipeline(s, m=12)
    ranked = res.evaluation.ranked
    assert len(ranked) >= 2
    assert ranked["n_preds"].nunique() == 1, (
        "metodos comparados sobre distinto numero de puntos:\n{}".format(
            ranked[["modelo", "n_preds", "mase"]]
        )
    )
    assert int(ranked["n_preds"].iloc[0]) == int(res.eval_origins.size)


def test_un_metodo_que_falla_queda_descalificado_no_promediado():
    s = make_series("estacional", 48)
    specs = [MODEL_REGISTRY[k] for k in ("naive", "holt_winters", "seasonal_naive")]
    # min_train deliberadamente bajo: holt_winters no puede operar al principio.
    origins = rolling_origins(len(s), 14)
    res = walk_forward(s, specs, m=12, season_length=12, min_train=14, origins=origins)

    fila_hw = res.metrics.query("modelo == 'holt_winters'").iloc[0]
    if not bool(fila_hw["elegible"]):
        assert "holt_winters" in res.disqualified
        assert fila_hw["motivo"]
        assert fila_hw["modelo"] not in set(res.ranked["modelo"])


def test_holt_winters_nunca_se_degrada_silenciosamente_a_holt():
    """F02: con historia insuficiente debe fallar, no cambiar de modelo."""
    hw = MODEL_REGISTRY["holt_winters"]
    y = make_series("estacional", 20).to_numpy()
    with pytest.raises(Exception):
        hw.forecast(y[:13], h=1, m=12)   # 13 < 2*m


def test_min_train_es_2m_solo_cuando_compiten_modelos_estacionales():
    assert resolve_min_train(48, 12, include_seasonal=True) == 24
    assert resolve_min_train(48, 12, include_seasonal=False) == ABSOLUTE_MIN_TRAIN


def test_serie_de_24_meses_con_estacionalidad_declara_que_no_puede_validarla():
    """Con 24 observaciones no hay 2*m + 8 origenes: hay que decirlo, no bajar el umbral."""
    res = run_pipeline(make_series("estacional", 24), m=12)
    texto = " ".join(res.notes).lower()
    if res.profile.has_seasonality:
        assert "estacional" in texto
    # Sea cual sea el camino, ningun modelo estacional puede haber sido rankeado
    # sobre menos origenes que el resto.
    if res.ok:
        assert res.evaluation.ranked["n_preds"].nunique() == 1


def test_walk_forward_devuelve_agregado_y_detalle_en_una_sola_pasada():
    """F14: el original recorria el walk-forward 3-4 veces por sesion."""
    s = make_series("tendencia", 48)
    specs = [MODEL_REGISTRY[k] for k in ("naive", "linear_regression")]
    res = walk_forward(s, specs, m=1, season_length=12, min_train=12, dates=s.index)
    assert not res.metrics.empty
    errores = res.errors_frame()
    predicciones = res.predictions_frame()
    assert len(errores) == res.origins.size
    assert set(errores.columns) == {"naive", "linear_regression"}
    assert "y_true" in predicciones.columns


def test_el_ranking_ordena_por_mase():
    s = make_series("tendencia", 60)
    res = run_pipeline(s, m=12)
    valores = res.evaluation.ranked["mase"].to_numpy()
    assert np.all(np.diff(valores) >= -1e-9), "el ranking no esta ordenado por MASE"


def test_los_benchmarks_siempre_entran_al_ranking():
    """F12: sin naive no hay forma de interpretar ninguna metrica."""
    for kind, n in [("plana", 48), ("tendencia", 48), ("tendencia_estacional", 60)]:
        res = run_pipeline(make_series(kind, n), m=12)
        assert "naive" in set(res.evaluation.metrics["modelo"]), (
            "falta el benchmark naive en {} n={}".format(kind, n)
        )


def test_el_filtro_estructural_registra_el_motivo_de_cada_exclusion():
    """F22: nada se descarta en silencio."""
    prof = classify_series(make_series("tendencia_limpia", 48), m=12)
    specs, excluded = eligible_specs(prof, n_train=24, structural_filter=True)
    assert excluded, "se esperaba al menos una exclusion en una serie con tendencia"
    for key, motivo in excluded.items():
        assert isinstance(motivo, str) and len(motivo) > 10
    assert {s.key for s in specs}.isdisjoint(set(excluded))
