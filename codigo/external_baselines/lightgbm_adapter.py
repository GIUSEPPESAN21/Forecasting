"""Adaptador LightGBM como modelo GLOBAL vía `mlforecast` (Fase 11, F27).

Import perezoso: `mlforecast`/`lightgbm` solo se importan dentro de
`fit_predict_lgbm`, nunca a nivel de módulo (mismo criterio que
`prophet_adapter.py`).

Por qué `mlforecast` y no LightGBM "a mano"
--------------------------------------------
`mlforecast` es de la misma familia Nixtla que `statsforecast`, ya integrado
en `forecasting_core` (F05/F13) — mantiene consistencia de ecosistema sin
agregar una dependencia conceptualmente nueva. Resuelve la construcción de
features de lag/rolling y el pronóstico recursivo multi-paso, que es
exactamente lo que un modelo "global" de gradient boosting necesita para
competir en el mismo protocolo de un paso adelante que usa el resto de este
proyecto (Hyndman & Athanasopoulos 2021, cap. sobre modelos globales;
LightGBM fue el modelo base de la mayoría de las soluciones ganadoras de la
competencia M5).

Sin ajuste fino: parámetros de LightGBM elegidos únicamente para que el
modelo no falle por falta de datos en series cortas (n=24-48), no para
maximizar precisión — es una línea base de comparación, no una tesis sobre
tuning de gradient boosting (ver §6 del prompt de la Fase 11).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["fit_predict_lgbm", "LGBM_MIN_OBS"]

# Con menos de esto no queda suficiente historia para ni siquiera el lag mas
# corto (1) mas un puñado de filas de entrenamiento tras eliminar NaN.
LGBM_MIN_OBS = 15

_LAG_CANDIDATES = (1, 2, 3, 6, 12)
_ROLLING_CANDIDATES = (3, 6, 12)


def _select_lags(n: int) -> list[int]:
    """Lags estandar de mlforecast, acotados para no vaciar el train (F13-like)."""
    floor = max(4, n // 2)
    lags = [l for l in _LAG_CANDIDATES if l < floor]
    return lags or [1]


def _select_rolling_windows(n: int) -> list[int]:
    floor = max(3, n // 3)
    return [w for w in _ROLLING_CANDIDATES if w < floor]


def fit_predict_lgbm(
    y,
    h: int,
    freq: str = "MS",
    *,
    clip_non_negative: bool = True,
    random_state: int = 0,
) -> np.ndarray:
    """Ajusta un LightGBM global (via `mlforecast`) y devuelve `h` pasos.

    Igual que `fit_predict_prophet`, construye un indice de fechas sintetico
    a partir de `freq`: solo el espaciado importa, no las fechas reales.
    """
    try:
        from lightgbm import LGBMRegressor
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import RollingMean
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "mlforecast/lightgbm no estan instalados. Instale con: "
            "pip install -r requirements-external.txt"
        ) from exc

    yv = np.asarray(y, dtype=float).ravel()
    yv = yv[np.isfinite(yv)]
    n = yv.size
    if n < LGBM_MIN_OBS:
        raise ValueError(
            "lightgbm/mlforecast requiere al menos {} observaciones, recibio {}".format(
                LGBM_MIN_OBS, n
            )
        )
    h = int(h)
    if h < 1:
        raise ValueError("h debe ser >= 1")

    lags = _select_lags(n)
    windows = _select_rolling_windows(n)
    lag_transforms = (
        {1: [RollingMean(window_size=w) for w in windows]} if windows else None
    )
    logger.info("lgbm: n=%d lags=%s rolling_windows=%s", n, lags, windows)

    idx = pd.date_range("2000-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"unique_id": "serie", "ds": idx, "y": yv})

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )
    fcst = MLForecast(models={"lgbm": model}, freq=freq, lags=lags, lag_transforms=lag_transforms)
    try:
        fcst.fit(df, static_features=[])
    except Exception as exc:
        raise ValueError(
            "mlforecast/lightgbm fallo al ajustar (n={}, lags={}): {}: {}".format(
                n, lags, type(exc).__name__, exc
            )
        ) from exc

    preds = fcst.predict(h)
    out = preds["lgbm"].to_numpy(dtype=float)

    if out.size != h:
        raise ValueError("lightgbm devolvio {} valores para h={}".format(out.size, h))

    if clip_non_negative:
        # Mismo criterio que `ModelSpec.forecast(..., clip_non_negative=True)`
        # del nucleo (F21), duplicado a proposito (ver docstring del paquete).
        out = np.clip(out, 0.0, None)

    return out
