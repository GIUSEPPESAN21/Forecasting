"""F05 - la metrica reportada nunca es el minimo de un barrido.

El defecto original: se elegia el metodo por walk-forward sobre TODA la serie,
se barrian hasta 361 combinaciones minimizando ese mismo MAPE, y se reportaba el
minimo resultante como desempeno. Con 18 puntos de evaluacion y 361 candidatos,
ese minimo esta dominado por ruido de seleccion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_core.models import MODEL_REGISTRY
from forecasting_core.optimize import (
    MIN_TUNE_ORIGINS, honest_outer_estimate, refine_around, run_pipeline, tune_spec,
)
from forecasting_core.validation import walk_forward
from conftest import make_series


def test_los_bloques_de_tuning_y_evaluacion_son_disjuntos():
    res = run_pipeline(make_series("tendencia_estacional", 72), m=12)
    tune = set(res.tune_origins.tolist())
    ev = set(res.eval_origins.tolist())
    assert tune and ev
    assert tune.isdisjoint(ev), "el bloque de evaluacion participo en el tuning"
    assert max(tune) < min(ev), "los bloques deben ser contiguos y en orden temporal"


def test_el_bloque_de_evaluacion_es_el_final_de_la_serie():
    """Evaluar sobre el tramo mas reciente es lo que importa operativamente."""
    res = run_pipeline(make_series("tendencia", 60), m=12)
    assert int(res.eval_origins[-1]) == int(res.origins[-1])


def test_el_mase_reportado_no_es_el_minimo_del_barrido():
    """La prueba central de F05."""
    s = make_series("tendencia_estacional", 72)
    res = run_pipeline(s, m=12)
    assert res.ok

    afinados = [k for k, t in res.tuning.items() if t.tuned and t.n_evaluated > 1]
    assert afinados, "ningun modelo fue afinado: la prueba no verifica nada"

    for key in afinados:
        tun = res.tuning[key]
        fila = res.evaluation.metrics.query("modelo == @key")
        if fila.empty or not bool(fila.iloc[0]["elegible"]):
            continue
        reportado = float(fila.iloc[0]["mase"])
        minimo_barrido = float(tun.tune_mase)
        # El MASE reportado se mide en un bloque que el barrido nunca vio, asi
        # que coincidir con el minimo del barrido seria una casualidad enorme.
        assert reportado != pytest.approx(minimo_barrido, rel=1e-9), (
            "{}: el MASE reportado ({:.6f}) es identico al minimo del barrido "
            "({:.6f})".format(key, reportado, minimo_barrido)
        )


def test_el_tuning_solo_ve_el_bloque_de_tuning():
    """Cambiar el bloque de evaluacion no puede alterar los parametros elegidos."""
    s = make_series("tendencia", 60)
    y = s.to_numpy(dtype=float)
    spec = MODEL_REGISTRY["ses"]
    origins = np.arange(12, 50)
    tune_origins = origins[:-10]
    scale = y[: origins[0]]

    a = tune_spec(y, spec, m=1, season_length=12, tune_origins=tune_origins,
                  scale_train=scale, n_obs=len(y))

    contaminado = y.copy()
    contaminado[tune_origins[-1] + 1:] *= 3.0    # se altera solo el bloque eval
    b = tune_spec(contaminado, spec, m=1, season_length=12, tune_origins=tune_origins,
                  scale_train=scale, n_obs=len(y))

    assert a.params == b.params, (
        "los parametros cambiaron al alterar el bloque de evaluacion: hay fuga"
    )


def test_si_no_hay_bloque_de_tuning_se_usan_los_valores_por_defecto_y_se_declara():
    """Nunca afinar sobre el mismo bloque que se reporta, ni siquiera con poca data."""
    res = run_pipeline(make_series("plana", 24), m=12, eval_block=10)
    if res.tune_origins.size < MIN_TUNE_ORIGINS:
        for key, tun in res.tuning.items():
            if MODEL_REGISTRY[key].has_hyperparameters:
                assert tun.tuned is False
                assert "insuficiente" in tun.reason or "defecto" in tun.reason
        assert any("defecto" in n or "hiperparametros" in n for n in res.notes)


def test_la_parrilla_refinada_esta_acotada():
    """F13: coarse-to-fine no puede volver a explotar."""
    assert len(refine_around({"alpha": 0.5})) <= 5
    assert len(refine_around({"alpha": 0.5, "beta": 0.3})) <= 25
    assert len(refine_around({"alpha": 0.5, "beta": 0.3, "gamma": 0.1})) <= 27


def test_refine_respeta_los_limites():
    for cand in refine_around({"alpha": 0.02}) + refine_around({"alpha": 0.98}):
        assert 0.0 < cand["alpha"] < 1.0


def test_estimacion_externa_honesta_reserva_un_bloque_intacto():
    """El metodo TAMBIEN se elige a ciegas: es el numero que va al manuscrito."""
    s = make_series("tendencia_estacional", 84)
    out = honest_outer_estimate(s, m=12, outer_block=6)
    assert out["ok"], out.get("reason")
    assert out["n_outer"] == 6
    metricas = out["outer_metrics"]
    assert not metricas.empty
    assert "naive" in set(metricas["modelo"]), "falta el benchmark en el bloque externo"
    # El bloque externo nunca aparece entre los origenes del pipeline interno.
    interno = out["inner"]
    n = len(s)
    assert int(interno.origins.max()) < n - 6


def test_estimacion_externa_no_duplica_el_modelo_si_el_ganador_es_naive():
    """Regresion: si el ganador interno ES naive/seasonal_naive, el bloque
    externo evaluaba dos filas 'naive', y cualquier .loc['naive'] aguas abajo
    (panel_publico.py, vs_incumbente.py) lanzaba ValueError de ambiguedad."""
    rng = np.random.default_rng(2)  # semilla que produce winner='naive'
    idx = pd.date_range("2020-01-01", periods=42, freq="MS")
    s = pd.Series(1000 + rng.normal(0, 80, 42), index=idx)
    out = honest_outer_estimate(s, m=12, outer_block=6)
    assert out["ok"]
    metricas = out["outer_metrics"]
    assert metricas["modelo"].value_counts().max() == 1, (
        "modelo duplicado en el bloque externo: {}".format(
            metricas["modelo"].value_counts().to_dict()
        )
    )


def test_estimacion_externa_se_niega_si_la_serie_es_muy_corta():
    out = honest_outer_estimate(make_series("plana", 24), m=12, outer_block=6)
    assert out["ok"] is False
    assert "corta" in out["reason"]


def test_todos_los_modelos_se_evaluan_con_los_mismos_origenes_tras_afinar():
    """Afinar no puede darle a un modelo mas puntos de evaluacion que a otro."""
    res = run_pipeline(make_series("tendencia_estacional", 72), m=12)
    n_esperado = int(res.eval_origins.size)
    for _, fila in res.evaluation.ranked.iterrows():
        assert int(fila["n_preds"]) == n_esperado
