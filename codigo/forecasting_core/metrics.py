"""Metricas de error de pronostico.

Resuelve F06 (ME ausente) y F12 (MAPE explota con demanda cero, sin MASE).

Principio: MASE es la metrica primaria de ranking (escala-independiente, definida
en cero, comparable entre series). MAPE se conserva como metrica secundaria de
comunicacion, excluyendo explicitamente los periodos con demanda cero y
reportando cuantos se excluyeron.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

__all__ = [
    "mse", "rmse", "mad", "me", "mape", "smape", "mase", "tracking_signal",
    "MetricSet", "compute_metrics", "seasonal_naive_scale",
]

# Umbral por debajo del cual una demanda se considera cero para efectos de MAPE.
ZERO_TOL = 1e-9


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Alinea y filtra pares con prediccion no finita."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} y y_pred {yp.shape} no tienen la misma forma")
    mask = np.isfinite(yp) & np.isfinite(yt)
    return yt[mask], yp[mask]


def mse(y_true, y_pred) -> float:
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred) -> float:
    v = mse(y_true, y_pred)
    return float(np.sqrt(v)) if np.isfinite(v) else float("nan")


def mad(y_true, y_pred) -> float:
    """Mean Absolute Deviation. CON valor absoluto (ver F23: la tesis lo omite)."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def me(y_true, y_pred) -> float:
    """Mean Error (sesgo). F06: declarado en ambos manuscritos, nunca implementado.

    Signo positivo => el modelo subestima la demanda (riesgo de faltante).
    Signo negativo => el modelo sobreestima (riesgo de sobrestock).
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yt - yp))


def mape(y_true, y_pred, return_excluded: bool = False):
    """MAPE excluyendo periodos con demanda cero (F12).

    El codigo original usaba denom = max(|y|, 1e-8), lo que no protege: convierte
    un cero en un error de 1e10 %. Aqui los ceros se excluyen del promedio y se
    informa cuantos fueron.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    nz = np.abs(yt) > ZERO_TOL
    n_excluded = int((~nz).sum())
    if nz.sum() == 0:
        val = float("nan")
    else:
        val = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0)
    if return_excluded:
        return val, n_excluded
    return val


def smape(y_true, y_pred) -> float:
    """sMAPE simetrico (Makridakis 1993), definido tambien cuando y=0."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    ok = denom > ZERO_TOL
    if ok.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(yt[ok] - yp[ok]) / denom[ok]) * 100.0)


def seasonal_naive_scale(y_train: Sequence[float], m: int = 1) -> float:
    """Denominador de MASE: MAE in-sample del naive estacional sobre el train.

    Hyndman & Koehler (2006). Si m > 1 y no hay suficientes datos, cae a m=1
    (naive simple) de forma EXPLICITA, no silenciosa: devuelve nan si tampoco
    hay datos para m=1, y quien llama debe decidir.
    """
    yv = np.asarray(y_train, dtype=float).ravel()
    yv = yv[np.isfinite(yv)]
    eff_m = m if (m >= 1 and yv.size > m) else 1
    if yv.size <= eff_m:
        return float("nan")
    diffs = np.abs(yv[eff_m:] - yv[:-eff_m])
    scale = float(np.mean(diffs))
    return scale if scale > ZERO_TOL else float("nan")


def mase(y_true, y_pred, y_train, m: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006). Metrica PRIMARIA de ranking.

    Definicion exacta usada en todo el pipeline (F35: la Ecuacion 1 del
    manuscrito fue corregida en la Fase 14 para reflejar este mismo
    escalado sobre el bloque de entrenamiento):

        MASE = MAD(y_true, y_pred) / scale

        scale = MAE in-sample del naive estacional, calculado SOLO sobre el
                bloque de ENTRENAMIENTO (`y_train`, llamado `scale_train` en
                `optimize.py`/`validation.py`: `y[:origins[0]]`, es decir las
                observaciones ANTERIORES al primer origen de evaluacion) -
                nunca sobre la serie completa ni sobre el bloque de
                evaluacion. `scale = mean(|y_train[m:] - y_train[:-m]|)`.

        m = 12 si `classify_series` confirmo estacionalidad, m = 1 en caso
            contrario (ver `seasonal_naive_scale`). No es un parametro libre:
            lo fija la clasificacion estructural de la propia serie.

    MASE < 1  => el modelo evaluado mejora al naive estacional in-sample
                 medido en el bloque de entrenamiento.
    MASE >= 1 => no aporta sobre repetir el ultimo valor (o el de hace m
                 periodos).
    """
    scale = seasonal_naive_scale(y_train, m=m)
    if not np.isfinite(scale):
        return float("nan")
    num = mad(y_true, y_pred)
    if not np.isfinite(num):
        return float("nan")
    return float(num / scale)


def tracking_signal(y_true, y_pred) -> float:
    """Senal de rastreo = suma acumulada del error / MAD.

    Regla operativa clasica: |TS| > 4 indica sesgo sistematico que exige
    reajustar el modelo (Nahmias & Olsen, 2015).
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    d = mad(yt, yp)
    if not np.isfinite(d) or d <= ZERO_TOL:
        return float("nan")
    return float(np.sum(yt - yp) / d)


@dataclass(frozen=True)
class MetricSet:
    """Conjunto completo de metricas de un metodo sobre un bloque de evaluacion."""
    n_preds: int
    mase: float
    mape: float
    mape_n_excluded: int
    mad: float
    mse: float
    rmse: float
    me: float
    smape: float
    tracking_signal: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true, y_pred, y_train, m: int = 1) -> MetricSet:
    """Calcula todas las metricas de una vez sobre un mismo par (y_true, y_pred)."""
    yt, yp = _clean_pair(y_true, y_pred)
    mp, n_exc = mape(y_true, y_pred, return_excluded=True)
    return MetricSet(
        n_preds=int(yt.size),
        mase=mase(y_true, y_pred, y_train, m=m),
        mape=mp,
        mape_n_excluded=n_exc,
        mad=mad(y_true, y_pred),
        mse=mse(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        me=me(y_true, y_pred),
        smape=smape(y_true, y_pred),
        tracking_signal=tracking_signal(y_true, y_pred),
    )
