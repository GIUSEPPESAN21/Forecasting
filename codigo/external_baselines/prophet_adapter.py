"""Adaptador Prophet (Fase 11, F26/F27).

Import perezoso: `prophet`/`cmdstanpy` solo se importan dentro de
`fit_predict_prophet`, nunca a nivel de módulo, para que este paquete pueda
importarse (y sus tests puedan colectarse y saltarse con
`pytest.importorskip`) sin Prophet instalado.

Sobre F26 (por qué `yearly_seasonality` no se activa siempre)
---------------------------------------------------------------
El PDF `comparacion_herramientas.pdf` aportado por los tutores corrió Prophet
con `yearly_seasonality` activada de forma incondicional, incluida la serie de
n=24 observaciones (exactamente 2 ciclos anuales, el mínimo teórico para que
un componente Fourier anual esté identificado). Resultado documentado ahí:
MAPE=353.98% y pronósticos negativos. Ese no es un fallo de Prophet en
abstracto — es un fallo de configuración: la propia documentación de Prophet
recomienda desactivar componentes estacionales que la serie no puede
sostener. Aquí la regla es explícita y se aplica siempre, no solo cuando
"se acuerda": `yearly_seasonality` se activa si y solo si hay al menos
`PROPHET_MIN_OBS_YEARLY` (2 ciclos anuales completos) observaciones; con
menos, se desactiva y la decisión queda registrada en el log.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "fit_predict_prophet",
    "PROPHET_MIN_OBS",
    "PROPHET_MIN_OBS_YEARLY",
]

# Piso absoluto: por debajo de esto un ajuste bayesiano no tiene sentido
# (menos que el mínimo del propio núcleo para cualquier análisis, F15).
PROPHET_MIN_OBS = 10
# Dos ciclos anuales completos (2 * 12): mínimo para que un componente de
# Fourier de periodo 12 esté identificado en vez de absorber ruido (F26).
PROPHET_MIN_OBS_YEARLY = 24


def _quiet_prophet_logging() -> None:
    """Prophet/cmdstanpy son ruidosos por defecto (INFO en cada ajuste).

    No se usa `warnings.filterwarnings("ignore")` global (violaría F16):
    solo se sube el nivel de LOS loggers de estas dos librerías puntuales.
    """
    for name in ("prophet", "cmdstanpy"):
        logging.getLogger(name).setLevel(logging.WARNING)


def fit_predict_prophet(
    y,
    h: int,
    freq: str = "MS",
    *,
    yearly_seasonality: bool | None = None,
    clip_non_negative: bool = True,
) -> np.ndarray:
    """Ajusta Prophet sobre `y` y devuelve el pronóstico puntual de `h` pasos.

    `y` es un array/serie sin fechas: aquí se construye un índice de fechas
    sintético (`freq`, por defecto mensual) porque Prophet exige una columna
    `ds`; el índice real de la serie del usuario no importa para el ajuste,
    solo su frecuencia y espaciado, que este adaptador siempre recibe como
    mensual desde `external_baselines.adapters`.

    `yearly_seasonality=None` (default) aplica la regla de F26: se activa
    solo con `n >= PROPHET_MIN_OBS_YEARLY`. Pasar `True`/`False` explícito la
    sobreescribe (usado solo en tests que documentan el contraste).
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "prophet no esta instalado. Instale con: "
            "pip install -r requirements-external.txt"
        ) from exc

    _quiet_prophet_logging()

    yv = np.asarray(y, dtype=float).ravel()
    yv = yv[np.isfinite(yv)]
    n = yv.size
    if n < PROPHET_MIN_OBS:
        raise ValueError(
            "prophet requiere al menos {} observaciones, recibio {}".format(
                PROPHET_MIN_OBS, n
            )
        )
    h = int(h)
    if h < 1:
        raise ValueError("h debe ser >= 1")

    if yearly_seasonality is None:
        yearly_seasonality = n >= PROPHET_MIN_OBS_YEARLY
        logger.info(
            "yearly_seasonality=%s decidido automaticamente (n=%d, umbral F26=%d)",
            yearly_seasonality, n, PROPHET_MIN_OBS_YEARLY,
        )

    idx = pd.date_range("2000-01-01", periods=n, freq=freq)
    df = pd.DataFrame({"ds": idx, "y": yv})

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=h, freq=freq, include_history=False)
    forecast = model.predict(future)
    out = forecast["yhat"].to_numpy(dtype=float)

    if out.size != h:
        raise ValueError("prophet devolvio {} valores para h={}".format(out.size, h))

    if clip_non_negative:
        # Mismo criterio que `ModelSpec.forecast(..., clip_non_negative=True)`
        # del nucleo (F21): se duplica esta unica linea a proposito en vez de
        # importar forecasting_core desde un modulo que debe seguir
        # funcionando de forma completamente aislada.
        out = np.clip(out, 0.0, None)

    return out
