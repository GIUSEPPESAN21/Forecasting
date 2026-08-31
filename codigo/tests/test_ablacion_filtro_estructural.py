"""F33 - la ablacion debe evaluar EXACTAMENTE las mismas series en ambos modos.

No se descarga M3 aqui (seria una prueba de integracion lenta y con
dependencia de red); se verifica la propiedad que realmente importa para que
la comparacion on/off sea valida: `run_config` recibe la misma lista de
`ids` y el mismo `df` sin importar `structural_filter`, y no descarta ni
reordena series entre una corrida y otra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos"))

import ablacion_filtro_estructural as afe  # noqa: E402


def test_run_config_evalua_las_mismas_series_en_ambos_modos(monkeypatch):
    ids = ["s1", "s2", "s3"]
    df = pd.DataFrame({
        "unique_id": ["s1"] * 5 + ["s2"] * 5 + ["s3"] * 5,
        "y": np.arange(15, dtype=float),
    })

    llamados = []

    def fake_evaluate_one(uid, y, structural_filter=True, **kw):
        llamados.append((uid, structural_filter))
        return {"unique_id": uid, "estado": "ok", "mase_ganador": 1.0, "mase_naive": 1.0}

    monkeypatch.setattr(afe, "evaluate_one", fake_evaluate_one)

    res_on, _ = afe.run_config(ids, df, structural_filter=True)
    res_off, _ = afe.run_config(ids, df, structural_filter=False)

    assert sorted(res_on["unique_id"]) == sorted(res_off["unique_id"]) == sorted(ids)
    ids_on = {u for u, sf in llamados if sf is True}
    ids_off = {u for u, sf in llamados if sf is False}
    assert ids_on == ids_off == set(ids)
