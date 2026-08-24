"""Fixtures compartidas. Toda serie sintetica usa semilla fija."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

SEED = 20260824


def make_series(kind: str, n: int, seed: int = SEED) -> pd.Series:
    """Series de control usadas por ambas auditorias, reproducibles."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    if kind == "plana":
        y = 1000 + rng.normal(0, 90, n)
    elif kind == "tendencia":
        y = 1000 + 25 * t + rng.normal(0, 90, n)
    elif kind == "tendencia_limpia":
        y = 1000 + 25 * t + rng.normal(0, 10, n)
    elif kind == "tendencia_perfecta":
        y = 1000 + 25 * t
    elif kind == "estacional":
        y = 1000 + 350 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 90, n)
    elif kind == "tendencia_estacional":
        y = 1000 + 20 * t + 350 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 90, n)
    elif kind == "paseo_aleatorio":
        y = 1000 + np.cumsum(rng.normal(0, 60, n))
    elif kind == "intermitente":
        y = rng.poisson(2.0, n) * rng.binomial(1, 0.45, n) * 50.0
    else:
        raise ValueError("tipo de serie desconocido: {}".format(kind))
    return pd.Series(np.asarray(y, dtype=float), index=idx, name="demand")


@pytest.fixture
def series_factory():
    return make_series


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)
