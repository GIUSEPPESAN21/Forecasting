"""F34 - desglose victorias/empates/derrotas, contra un caso resuelto a mano."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experimentos"))

from vs_incumbente import wtl_breakdown  # noqa: E402


def test_wtl_breakdown_caso_conocido_a_mano():
    # 6 filas, la ultima con NaN se excluye -> n=5 pares validos.
    # fila0: 0.5<1.0 victoria | fila1: 0.8==0.8 empate | fila2: 1.0==1.0 empate
    # fila3: 1.2>1.0 derrota  | fila4: 0.9==0.9 empate
    df = pd.DataFrame({
        "herramienta_mase": [0.5, 0.8, 1.0, 1.2, 0.9, np.nan],
        "naive_mase":       [1.0, 0.8, 1.0, 1.0, 0.9, 1.1],
    })
    r = wtl_breakdown(df, "herramienta_mase", "naive_mase")
    assert r == {"n": 5, "victorias": 1, "empates": 3, "derrotas": 1}


def test_wtl_breakdown_todo_victorias():
    df = pd.DataFrame({"a": [0.5, 0.6, 0.7], "b": [1.0, 1.0, 1.0]})
    r = wtl_breakdown(df, "a", "b")
    assert r == {"n": 3, "victorias": 3, "empates": 0, "derrotas": 0}


def test_wtl_breakdown_vacio():
    df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    r = wtl_breakdown(df, "a", "b")
    assert r == {"n": 0, "victorias": 0, "empates": 0, "derrotas": 0}
