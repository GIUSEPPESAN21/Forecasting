"""F32 - mase_naive debe poblarse incluso cuando el ganador ES naive.

Defecto original en `panel_publico.py::evaluate_one`: el bucle
`for metodo in ("naive", "seasonal_naive", out["winner"])` escribia la
columna "mase_{ganador o metodo}" con el mismo nombre "mase_ganador" tanto
para el iterador `metodo == winner` como para el propio `winner` al final,
de modo que si `winner == "naive"` la clave "mase_naive" nunca se escribia
-quedaba NaN- y esas filas se descartaban silenciosamente del test de
Wilcoxon (17 de 150 series en la corrida original).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos"))

from panel_publico import evaluate_one  # noqa: E402


def _serie_plana(n: int = 30, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1000.0 + rng.normal(0, 5, n)


def test_mase_naive_se_escribe_cuando_naive_es_el_ganador(monkeypatch):
    outer_metrics = pd.DataFrame([
        {"modelo": "naive", "mase": 0.95, "mape": 10.0, "me": 1.0, "elegible": True},
        {"modelo": "seasonal_naive", "mase": 1.10, "mape": 12.0, "me": 2.0, "elegible": True},
    ])
    fake_out = {"ok": True, "winner": "naive", "outer_metrics": outer_metrics}
    monkeypatch.setattr("panel_publico.honest_outer_estimate", lambda *a, **k: fake_out)

    row = evaluate_one("serie_test", _serie_plana())

    assert row["estado"] == "ok"
    assert row["ganador"] == "naive"
    assert "mase_naive" in row, "mase_naive nunca se escribio (F32 reaparecio)"
    assert np.isfinite(row["mase_naive"])
    assert row["mase_naive"] == pytest.approx(0.95)
    assert row["mase_ganador"] == pytest.approx(0.95)
    assert "mase_seasonal_naive" in row
    assert row["mase_seasonal_naive"] == pytest.approx(1.10)


def test_mase_naive_y_mase_ganador_son_columnas_distintas_cuando_gana_otro(monkeypatch):
    outer_metrics = pd.DataFrame([
        {"modelo": "naive", "mase": 1.20, "mape": 15.0, "me": 3.0, "elegible": True},
        {"modelo": "seasonal_naive", "mase": 1.05, "mape": 11.0, "me": 1.5, "elegible": True},
        {"modelo": "ses", "mase": 0.80, "mape": 8.0, "me": 0.5, "elegible": True},
    ])
    fake_out = {"ok": True, "winner": "ses", "outer_metrics": outer_metrics}
    monkeypatch.setattr("panel_publico.honest_outer_estimate", lambda *a, **k: fake_out)

    row = evaluate_one("serie_test", _serie_plana())

    assert row["ganador"] == "ses"
    assert row["mase_naive"] == pytest.approx(1.20)
    assert row["mase_ganador"] == pytest.approx(0.80)
    assert row["mase_naive"] != row["mase_ganador"]
