"""Intervalos de prediccion (F20).

La herramienta original entregaba exclusivamente pronosticos puntuales, pese a
que el abstract del manuscrito construia la contribucion sobre *uncertainty-aware
visual analytics*. Sin distribucion del error no se puede dimensionar stock de
seguridad: la herramienta entregaba la mitad del insumo que necesita un sistema
de inventarios.

Metodo
------
Se usan **cuantiles empiricos del error de pronostico por horizonte**, obtenidos
con origen rodante sobre la propia serie. Frente a los intervalos analiticos
tiene tres ventajas para este caso de uso:

* funciona para TODOS los modelos, incluidos promedio movil y regresion, que no
  tienen una distribucion predictiva cerrada;
* no asume normalidad ni homocedasticidad de los residuos;
* mide el error del procedimiento completo tal como se ejecuta en produccion,
  no el error condicional a un modelo dado por verdadero.

El costo esta acotado: `n_origins` ajustes (10 por defecto), no O(n).

Cuando hay pocos origenes, la incertidumbre de los propios cuantiles empiricos
es alta; en ese caso se recurre a una aproximacion normal escalada con sqrt(h)
sobre la desviacion de los errores de un paso, y se declara asi en `method`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .models import ModelSpec
from .validation import multi_horizon_errors

logger = logging.getLogger(__name__)

__all__ = ["PredictionInterval", "prediction_interval", "MIN_ORIGINS_FOR_EMPIRICAL"]

# Origenes minimos para confiar en cuantiles empiricos en vez de la normal.
MIN_ORIGINS_FOR_EMPIRICAL = 8


@dataclass
class PredictionInterval:
    """Pronostico puntual con banda, por horizonte."""

    index: pd.DatetimeIndex
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigma: np.ndarray
    level: float
    method: str
    n_origins: int
    bias: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        pct = int(round(self.level * 100))
        return pd.DataFrame(
            {
                "pronostico": self.mean,
                "inferior_{}".format(pct): self.lower,
                "superior_{}".format(pct): self.upper,
                "sigma_error": self.sigma,
                "sesgo_estimado": self.bias,
            },
            index=self.index,
        )


def prediction_interval(
    series,
    spec: ModelSpec,
    params: dict | None,
    *,
    season_length: int = 12,
    horizon: int = 12,
    level: float = 0.95,
    n_origins: int = 10,
    min_train: int | None = None,
    clip_non_negative: bool = True,
    future_index: pd.DatetimeIndex | None = None,
) -> PredictionInterval:
    """Pronostico de `horizon` pasos con banda de prediccion empirica."""
    s = pd.Series(series).astype(float).dropna()
    y = s.to_numpy(dtype=float)

    mean = spec.forecast(
        y, params=params, h=horizon, m=season_length, clip_non_negative=clip_non_negative
    )

    if future_index is None:
        if isinstance(s.index, pd.DatetimeIndex):
            start = s.index[-1] + pd.offsets.MonthBegin(1)
            future_index = pd.date_range(start, periods=horizon, freq="MS")
        else:
            future_index = pd.RangeIndex(len(s), len(s) + horizon)

    errs = multi_horizon_errors(
        y, spec, params, season_length, horizon,
        n_origins=n_origins, min_train=min_train, clip_non_negative=clip_non_negative,
    )
    counts = np.isfinite(errs).sum(axis=0)
    usable = int(counts.min()) if counts.size else 0

    sigma = np.full(horizon, np.nan)
    bias = np.zeros(horizon)
    for h in range(horizon):
        col = errs[:, h]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            sigma[h] = float(np.std(col, ddof=1))
            bias[h] = float(np.mean(col))

    # Los horizontes sin datos suficientes se completan escalando con sqrt(h),
    # que es el crecimiento de la varianza de un paseo aleatorio.
    if np.isnan(sigma).any():
        base = sigma[np.isfinite(sigma)]
        s1 = float(base[0]) if base.size else float(np.std(np.diff(y), ddof=1))
        for h in range(horizon):
            if not np.isfinite(sigma[h]):
                sigma[h] = s1 * np.sqrt(h + 1)

    z = float(stats.norm.ppf(0.5 + level / 2.0))
    if usable >= MIN_ORIGINS_FOR_EMPIRICAL:
        lo_q, hi_q = (1 - level) / 2.0, 0.5 + level / 2.0
        lower = np.empty(horizon)
        upper = np.empty(horizon)
        for h in range(horizon):
            col = errs[:, h]
            col = col[np.isfinite(col)]
            if col.size >= MIN_ORIGINS_FOR_EMPIRICAL:
                lower[h] = mean[h] + float(np.quantile(col, lo_q))
                upper[h] = mean[h] + float(np.quantile(col, hi_q))
            else:
                lower[h] = mean[h] - z * sigma[h]
                upper[h] = mean[h] + z * sigma[h]
        method = "cuantiles empiricos del error por horizonte ({} origenes)".format(usable)
    else:
        lower = mean - z * sigma
        upper = mean + z * sigma
        method = (
            "aproximacion normal escalada con sqrt(h): solo {} origenes "
            "disponibles, insuficientes para cuantiles empiricos (minimo {})".format(
                usable, MIN_ORIGINS_FOR_EMPIRICAL
            )
        )
        logger.info("Intervalos por aproximacion normal: %s", method)

    if clip_non_negative:
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)

    return PredictionInterval(
        index=pd.DatetimeIndex(future_index), mean=mean, lower=lower, upper=upper,
        sigma=sigma, level=level, method=method, n_origins=usable, bias=bias,
    )
