"""Validacion temporal honesta: walk-forward de origen rodante.

Resuelve F02, F03 y F14.

* F02 - paridad. El codigo original promediaba MAPEs calculados sobre conjuntos
  de tamano distinto: "Holt-Winters" y "SARIMA" caian silenciosamente a Holt y
  ARIMA(1,1,1) en 8 de 18 origenes, y aun asi competian en el mismo ranking.
  Aqui `min_train` es global, un modelo que falla en un origen registra NaN
  explicito, y **un metodo con cualquier NaN queda excluido del ranking**: nunca
  se promedia sobre menos puntos que los demas.

* F03 - fuga in-sample. El grafico principal mostraba `_fitted_series`, cuyos
  valores ajustados de promedio movil incluian y[t] (`rolling(k).mean()`), y su
  MAPE se interpretaba como desempeno. Aqui la unica serie de referencia sobre
  el historico es la de pronosticos de un paso adelante fuera de muestra, que
  usa exactamente la misma funcion que el pronostico publicado.

* F14 - cómputo redundante. El original recorria el walk-forward tres o cuatro
  veces por sesion (`walk_forward_errors`, `walk_forward_detail` y la
  re-evaluacion del Modulo 3). Aqui hay una sola pasada que devuelve a la vez el
  agregado y el detalle por origen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics, seasonal_naive_scale
from .models import InsufficientHistory, ModelSpec

logger = logging.getLogger(__name__)

__all__ = [
    "BacktestResult", "WalkForwardResult", "rolling_origins", "resolve_min_train",
    "backtest_one_step", "walk_forward", "multi_horizon_errors",
    "MIN_ORIGINS", "ABSOLUTE_MIN_TRAIN",
]

# Minimo de origenes para que una metrica sea interpretable.
MIN_ORIGINS = 8
# Piso absoluto de entrenamiento, independiente de la estacionalidad.
ABSOLUTE_MIN_TRAIN = 10


def resolve_min_train(n: int, m: int, *, include_seasonal: bool) -> int:
    """min_train = max(10, 2*m) si compiten modelos estacionales, 10 si no.

    Usar 2*m=24 en una serie NO estacional dejaria sin origenes a una serie de
    24 meses; por eso `m` efectivo es 1 cuando la clasificacion no detecta
    estacionalidad (ver `SeriesProfile.seasonal_period`).
    """
    m_eff = int(m) if include_seasonal else 1
    return max(ABSOLUTE_MIN_TRAIN, 2 * m_eff)


def rolling_origins(n: int, min_train: int, horizon: int = 1) -> np.ndarray:
    """Indices t evaluados: se pronostica y[t] con y[:t]."""
    first = int(min_train)
    last = int(n) - int(horizon) + 1
    if last <= first:
        return np.array([], dtype=int)
    return np.arange(first, last, dtype=int)


@dataclass
class BacktestResult:
    """Backtest de un modelo: pronostico de un paso en cada origen."""

    key: str
    params: dict
    origins: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    failures: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True si el modelo produjo pronostico en TODOS los origenes."""
        return bool(np.isfinite(self.y_pred).all()) and self.y_pred.size > 0

    @property
    def n_preds(self) -> int:
        return int(np.isfinite(self.y_pred).sum())


def backtest_one_step(
    y: np.ndarray,
    spec: ModelSpec,
    params: dict | None,
    season_length: int,
    origins: np.ndarray,
    *,
    clip_non_negative: bool = True,
) -> BacktestResult:
    """Pronostico de un paso adelante en cada origen, sin mirar el futuro.

    En el origen t el modelo solo ve y[:t]. La prueba `tests/test_no_leakage.py`
    verifica esta propiedad de forma programatica para cada modelo del registro.
    """
    yv = np.asarray(y, dtype=float).ravel()
    preds = np.full(origins.size, np.nan, dtype=float)
    failures: list[str] = []
    for i, t in enumerate(origins):
        train = yv[:t]
        if not np.isfinite(train).all():
            failures.append("origen {}: la ventana de entrenamiento tiene NaN".format(t))
            continue
        try:
            preds[i] = spec.forecast(
                train, params=params, h=1, m=season_length,
                clip_non_negative=clip_non_negative,
            )[0]
        except InsufficientHistory as exc:
            failures.append("origen {}: {}".format(t, exc))
        except Exception as exc:
            failures.append("origen {}: {}: {}".format(t, type(exc).__name__, exc))
            logger.debug("%s fallo en el origen %s: %s", spec.key, t, exc)
    return BacktestResult(
        key=spec.key,
        params=dict(params or {}),
        origins=origins,
        y_true=yv[origins],
        y_pred=preds,
        failures=failures,
    )


@dataclass
class WalkForwardResult:
    """Resultado completo de una sola pasada de walk-forward."""

    origins: np.ndarray
    dates: pd.DatetimeIndex | None
    y_true: np.ndarray
    min_train: int
    m: int
    backtests: dict[str, BacktestResult]
    metrics: pd.DataFrame
    disqualified: dict[str, str] = field(default_factory=dict)

    def errors_frame(self) -> pd.DataFrame:
        """Errores (real - pronostico) por origen y metodo, para graficar."""
        data = {k: bt.y_true - bt.y_pred for k, bt in self.backtests.items()}
        idx = self.dates if self.dates is not None else self.origins
        return pd.DataFrame(data, index=idx)

    def predictions_frame(self) -> pd.DataFrame:
        data = {k: bt.y_pred for k, bt in self.backtests.items()}
        idx = self.dates if self.dates is not None else self.origins
        return pd.DataFrame(data, index=idx).assign(y_true=self.y_true)

    @property
    def ranked(self) -> pd.DataFrame:
        """Solo los metodos que compiten en igualdad de condiciones."""
        if self.metrics.empty:
            return self.metrics
        return self.metrics[self.metrics["elegible"]].reset_index(drop=True)

    @property
    def winner(self) -> str | None:
        r = self.ranked
        return None if r.empty else str(r.iloc[0]["modelo"])


def walk_forward(
    y,
    specs: list[ModelSpec],
    *,
    m: int,
    min_train: int,
    season_length: int = 12,
    params_by_key: dict[str, dict] | None = None,
    dates: pd.DatetimeIndex | None = None,
    origins: np.ndarray | None = None,
    scale_train: np.ndarray | None = None,
    clip_non_negative: bool = True,
) -> WalkForwardResult:
    """Una sola pasada: agregado y detalle por origen a la vez (F14).

    `m` es el periodo estacional de la SERIE y solo escala MASE (1 si no hay
    estacionalidad). `season_length` es el periodo que usan los MODELOS
    estacionales y es siempre 12 en datos mensuales: mezclarlos haria que
    seasonal_naive o Holt-Winters recibieran m=1 y se descalificaran solos.

    `scale_train` fija el denominador de MASE. Por defecto se usa y[:origins[0]],
    identico para todos los metodos, de modo que los MASE son comparables entre
    si y entre bloques (tune vs eval).
    """
    yv = np.asarray(pd.Series(y).astype(float).to_numpy()).ravel()
    n = yv.size
    params_by_key = params_by_key or {}
    if origins is None:
        origins = rolling_origins(n, min_train)

    if origins.size == 0:
        empty = pd.DataFrame(
            columns=["modelo", "etiqueta", "elegible", "motivo", "mase", "mape", "mad",
                     "mse", "rmse", "me", "smape", "tracking_signal", "n_preds", "params"]
        )
        return WalkForwardResult(origins, None, np.array([]), min_train, m, {}, empty)

    if scale_train is None:
        scale_train = yv[: origins[0]]
    scale = seasonal_naive_scale(scale_train, m=m)

    idx = None
    if dates is not None:
        idx = pd.DatetimeIndex(pd.Index(dates)[origins])

    backtests: dict[str, BacktestResult] = {}
    rows = []
    disqualified: dict[str, str] = {}
    n_origins = int(origins.size)

    for spec in specs:
        bt = backtest_one_step(
            yv, spec, params_by_key.get(spec.key), season_length, origins,
            clip_non_negative=clip_non_negative,
        )
        backtests[spec.key] = bt
        ms = compute_metrics(bt.y_true, bt.y_pred, scale_train, m=m)

        eligible = bt.complete
        reason = ""
        if not eligible:
            n_missing = n_origins - bt.n_preds
            reason = (
                "no produjo pronostico en {} de {} origenes ({}); excluido del "
                "ranking para no promediar sobre menos puntos que los demas".format(
                    n_missing, n_origins,
                    bt.failures[0] if bt.failures else "motivo no registrado",
                )
            )
            disqualified[spec.key] = reason
            logger.info("%s descalificado: %s", spec.key, reason)

        rows.append({
            "modelo": spec.key,
            "etiqueta": spec.label,
            "elegible": eligible,
            "motivo": reason,
            "mase": ms.mase,
            "mape": ms.mape,
            "mad": ms.mad,
            "mse": ms.mse,
            "rmse": ms.rmse,
            "me": ms.me,
            "smape": ms.smape,
            "tracking_signal": ms.tracking_signal,
            "n_preds": ms.n_preds,
            "mape_excluidos": ms.mape_n_excluded,
            "params": dict(bt.params),
        })

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        # Ranking primario por MASE (F12); MAD desempata. Los no elegibles van al
        # final con su motivo, visibles pero fuera de competencia.
        metrics = metrics.sort_values(
            ["elegible", "mase", "mad"], ascending=[False, True, True]
        ).reset_index(drop=True)
    if not np.isfinite(scale):
        logger.warning(
            "Escala de MASE no calculable (serie constante en el tramo de "
            "referencia); el ranking cae a MAD."
        )
        if not metrics.empty:
            metrics = metrics.sort_values(
                ["elegible", "mad"], ascending=[False, True]
            ).reset_index(drop=True)

    return WalkForwardResult(
        origins=origins,
        dates=idx,
        y_true=yv[origins],
        min_train=int(min_train),
        m=int(m),
        backtests=backtests,
        metrics=metrics,
        disqualified=disqualified,
    )


def multi_horizon_errors(
    y,
    spec: ModelSpec,
    params: dict | None,
    season_length: int,
    H: int,
    *,
    n_origins: int = 10,
    min_train: int | None = None,
    clip_non_negative: bool = True,
) -> np.ndarray:
    """Errores empiricos por horizonte, para intervalos de prediccion (F20).

    Devuelve una matriz (n_origins x H) de errores reales - pronosticados. El
    numero de origenes esta acotado (por defecto 10) para respetar el
    presupuesto de computo: el costo es O(n_origins) ajustes, no O(n).
    """
    yv = np.asarray(pd.Series(y).astype(float).dropna().to_numpy()).ravel()
    n = yv.size
    floor = (min_train if min_train is not None
             else max(ABSOLUTE_MIN_TRAIN, spec.min_obs(season_length)))
    last_origin = n - H
    if last_origin <= floor:
        # No hay historia para H pasos: se degrada a los horizontes que si caben.
        last_origin = max(floor + 1, n - 1)
    starts = np.unique(
        np.linspace(max(floor, last_origin - n_origins + 1), last_origin,
                    num=min(n_origins, max(1, last_origin - floor + 1)), dtype=int)
    )
    errs = np.full((starts.size, H), np.nan, dtype=float)
    for i, t in enumerate(starts):
        train = yv[:t]
        actual = yv[t : t + H]
        if actual.size == 0:
            continue
        try:
            pred = spec.forecast(
                train, params=params, h=H, m=season_length,
                clip_non_negative=clip_non_negative,
            )
        except Exception as exc:
            logger.debug("multi_horizon_errors: %s fallo en t=%s (%s)", spec.key, t, exc)
            continue
        errs[i, : actual.size] = actual - pred[: actual.size]
    return errs
