"""Registro explicito de modelos de pronostico.

Resuelve F04 y, por construccion, elimina la causa raiz de F03.

Dos decisiones de diseno gobiernan este modulo
----------------------------------------------
1. **Despacho por clave exacta.** El codigo original resolvia el metodo con
   `if "arima" in nombre.lower()`, y cualquier nombre sin coincidencia caia a un
   `return train[-1]` (paseo aleatorio). El resultado era que el "MAPE
   optimizado" de Promedio Simple/Movil/Ponderado era en realidad el de un
   modelo distinto. Aqui `get_spec()` levanta `KeyError` ante una clave
   desconocida: nunca se sustituye un modelo en silencio.

2. **Una sola funcion de pronostico por modelo.** El original tenia tres rutas
   distintas para el mismo metodo (`METHODS` para el ranking, `_make_predictor`
   para la optimizacion y `_fit_and_forecast`/`_fitted_series` para el
   resultado final), que discrepaban entre si. Aqui cada modelo expone una unica
   `forecast(train, params, h, m) -> array(h)`; el backtest de un paso es
   simplemente `h=1`. Es imposible que el modelo evaluado difiera del publicado.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

logger = logging.getLogger(__name__)

__all__ = [
    "ModelSpec", "MODEL_REGISTRY", "get_spec", "available_keys",
    "eligible_specs", "forecast_with", "STATSFORECAST_AVAILABLE",
]

try:  # statsforecast es opcional: el nucleo funciona sin el, con menos modelos.
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    STATSFORECAST_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    STATSFORECAST_AVAILABLE = False
    logger.warning(
        "statsforecast no disponible (%s): AutoARIMA/AutoETS/AutoTheta quedan "
        "fuera del registro.", _exc
    )


# ---------------------------------------------------------------------------
# Especificacion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    seasonal: bool
    constant_level: bool
    _forecast: Callable[[np.ndarray, dict, int, int], np.ndarray]
    default_params: dict = field(default_factory=dict)
    _grid: Callable[[int, int], list[dict]] | None = None
    _min_obs: Callable[[int], int] = lambda m: 4
    analytic_interval: bool = False
    note: str = ""

    def min_obs(self, m: int) -> int:
        return int(self._min_obs(m))

    def grid(self, n: int, m: int) -> list[dict]:
        if self._grid is None:
            return [dict(self.default_params)]
        return self._grid(n, m)

    @property
    def has_hyperparameters(self) -> bool:
        return self._grid is not None

    def forecast(
        self,
        train,
        params: dict | None = None,
        h: int = 1,
        m: int = 12,
        clip_non_negative: bool = True,
    ) -> np.ndarray:
        """Pronostico de h pasos. Ruta UNICA para backtest y resultado final.

        `clip_non_negative` aplica el piso en cero (F21) de forma identica en la
        validacion y en el pronostico publicado, para que la metrica reportada
        corresponda exactamente a lo que el usuario recibe.
        """
        y = np.asarray(train, dtype=float).ravel()
        y = y[np.isfinite(y)]
        need = self.min_obs(m)
        if y.size < need:
            raise InsufficientHistory(
                "{} requiere al menos {} observaciones, recibio {}".format(
                    self.key, need, y.size
                )
            )
        out = np.asarray(
            self._forecast(y, dict(self.default_params, **(params or {})), int(h), int(m)),
            dtype=float,
        ).ravel()
        if out.size != h:
            raise ValueError(
                "{} devolvio {} valores para h={}".format(self.key, out.size, h)
            )
        if clip_non_negative:
            out = np.maximum(out, 0.0)
        return out


class InsufficientHistory(ValueError):
    """El modelo no puede evaluarse con la historia disponible.

    Se levanta explicitamente en lugar de sustituir el modelo por otro (F02).
    """


# ---------------------------------------------------------------------------
# Implementaciones
# ---------------------------------------------------------------------------
def _f_naive(y, p, h, m):
    return np.repeat(y[-1], h)


def _f_seasonal_naive(y, p, h, m):
    if m < 2 or y.size < m:
        raise InsufficientHistory("seasonal_naive requiere al menos m={} observaciones".format(m))
    last_cycle = y[-m:]
    return np.array([last_cycle[i % m] for i in range(h)], dtype=float)


def _f_mean(y, p, h, m):
    return np.repeat(float(np.mean(y)), h)


def _f_moving_average(y, p, h, m):
    k = int(p.get("k", 3))
    if y.size < k:
        raise InsufficientHistory("moving_average k={} requiere {} observaciones".format(k, k))
    return np.repeat(float(np.mean(y[-k:])), h)


def _f_weighted_moving_average(y, p, h, m):
    w = np.asarray(p.get("weights", (0.5, 0.3, 0.2)), dtype=float)
    k = w.size
    if y.size < k:
        raise InsufficientHistory("weighted_moving_average requiere {} observaciones".format(k))
    w = w / w.sum()
    # w[0] pondera la observacion MAS reciente.
    return np.repeat(float(np.dot(y[-k:], w[::-1])), h)


def _f_linear_regression(y, p, h, m):
    n = y.size
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return intercept + slope * np.arange(n, n + h, dtype=float)


def _fit_ets(y, *, trend=None, seasonal=None, seasonal_periods=None, params=None):
    """Ajusta un ETS de statsmodels registrando cualquier degradacion.

    La inicializacion 'estimated' es preferible a 'heuristic' en series cortas;
    si falla se registra en el log y se reintenta con 'heuristic'. Es una
    degradacion de la INICIALIZACION, nunca una sustitucion del modelo.
    """
    params = params or {}
    kwargs = dict(trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    for init in ("estimated", "heuristic"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if seasonal is None and trend is None:
                    model = SimpleExpSmoothing(y, initialization_method=init)
                else:
                    model = ExponentialSmoothing(y, initialization_method=init, **kwargs)
                return model.fit(optimized=False, **params)
        except Exception as exc:
            if init == "heuristic":
                raise
            logger.debug("Init 'estimated' fallo (%s); se reintenta con 'heuristic'", exc)
    raise RuntimeError("unreachable")


def _f_ses(y, p, h, m):
    fit = _fit_ets(y, params={"smoothing_level": float(p.get("alpha", 0.2))})
    return np.asarray(fit.forecast(h), dtype=float)


def _f_holt(y, p, h, m):
    fit = _fit_ets(
        y,
        trend="add",
        params={
            "smoothing_level": float(p.get("alpha", 0.2)),
            "smoothing_trend": float(p.get("beta", 0.1)),
        },
    )
    return np.asarray(fit.forecast(h), dtype=float)


def _f_holt_winters(y, p, h, m):
    if m < 2 or y.size < 2 * m:
        raise InsufficientHistory(
            "holt_winters requiere al menos 2*m={} observaciones, recibio {}".format(
                2 * m, y.size
            )
        )
    fit = _fit_ets(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=m,
        params={
            "smoothing_level": float(p.get("alpha", 0.2)),
            "smoothing_trend": float(p.get("beta", 0.1)),
            "smoothing_seasonal": float(p.get("gamma", 0.1)),
        },
    )
    return np.asarray(fit.forecast(h), dtype=float)


def _sf_forecast(model, y, h):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.forecast(y=np.asarray(y, dtype=np.float32), h=int(h))
    return np.asarray(res["mean"], dtype=float)


def _f_auto_arima(y, p, h, m):
    model = AutoARIMA(season_length=1, seasonal=False, **p)
    return _sf_forecast(model, y, h)


def _f_auto_sarima(y, p, h, m):
    if m < 2 or y.size < 2 * m:
        raise InsufficientHistory(
            "auto_sarima requiere al menos 2*m={} observaciones, recibio {}".format(2 * m, y.size)
        )
    model = AutoARIMA(season_length=int(m), **p)
    return _sf_forecast(model, y, h)


def _f_auto_ets(y, p, h, m):
    model = AutoETS(season_length=int(m) if m >= 2 else 1, **p)
    return _sf_forecast(model, y, h)


def _f_auto_theta(y, p, h, m):
    model = AutoTheta(season_length=int(m) if m >= 2 else 1, **p)
    return _sf_forecast(model, y, h)


# ---------------------------------------------------------------------------
# Parrillas (acotadas; ARIMA/SARIMA ya no barren nada - F13)
# ---------------------------------------------------------------------------
# Las parrillas son GRUESAS a proposito: `optimize.refine_around` hace despues
# una busqueda local fina alrededor del mejor punto. Coarse-to-fine acota el
# costo a O(coarse + refine) en lugar de O(paso^-k), que era lo que hacia que
# Holt evaluara 361 combinaciones sobre 18 observaciones (F05, F13).
def _grid_ses(n, m):
    return [{"alpha": round(float(a), 3)} for a in np.arange(0.1, 0.91, 0.1)]


def _grid_holt(n, m):
    vals = (0.1, 0.3, 0.5, 0.7, 0.9)
    return [{"alpha": a, "beta": b} for a in vals for b in vals]


def _grid_holt_winters(n, m):
    coarse = (0.1, 0.4, 0.7)
    return [
        {"alpha": a, "beta": b, "gamma": g}
        for a in coarse
        for b in coarse
        for g in coarse
    ]


def _grid_moving_average(n, m):
    kmax = min(6, max(2, n // 3))
    return [{"k": k} for k in range(2, kmax + 1)]


def _grid_wma(n, m):
    return [
        {"weights": (0.5, 0.3, 0.2)},
        {"weights": (0.6, 0.25, 0.15)},
        {"weights": (0.4, 0.35, 0.25)},
        {"weights": (0.7, 0.2, 0.1)},
        {"weights": (0.5, 0.25, 0.15, 0.10)},
    ]


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------
def _build_registry() -> dict[str, ModelSpec]:
    specs = [
        ModelSpec(
            key="naive", label="Naive (ultimo valor)", family="benchmark",
            seasonal=False, constant_level=True, _forecast=_f_naive,
            _min_obs=lambda m: 2,
            note="Benchmark obligatorio. MASE se escala contra el naive estacional.",
        ),
        ModelSpec(
            key="seasonal_naive", label="Naive estacional (valor de hace m periodos)",
            family="benchmark", seasonal=True, constant_level=False,
            _forecast=_f_seasonal_naive, _min_obs=lambda m: max(2, m + 1),
            note="Benchmark obligatorio para series estacionales (F12).",
        ),
        ModelSpec(
            key="mean", label="Promedio simple", family="promedio",
            seasonal=False, constant_level=True, _forecast=_f_mean,
            _min_obs=lambda m: 3,
        ),
        ModelSpec(
            key="moving_average", label="Promedio movil", family="promedio",
            seasonal=False, constant_level=True, _forecast=_f_moving_average,
            default_params={"k": 3}, _grid=_grid_moving_average,
            _min_obs=lambda m: 7,
        ),
        ModelSpec(
            key="weighted_moving_average", label="Promedio movil ponderado",
            family="promedio", seasonal=False, constant_level=True,
            _forecast=_f_weighted_moving_average,
            default_params={"weights": (0.5, 0.3, 0.2)}, _grid=_grid_wma,
            _min_obs=lambda m: 5,
        ),
        ModelSpec(
            key="ses", label="Suavizamiento exponencial simple", family="suavizamiento",
            seasonal=False, constant_level=False, _forecast=_f_ses,
            default_params={"alpha": 0.2}, _grid=_grid_ses, _min_obs=lambda m: 5,
            note="Nivel ADAPTATIVO, no constante: sigue desplazamientos del nivel, "
                 "asi que no se excluye por el filtro de nivel estable.",
        ),
        ModelSpec(
            key="holt", label="Holt (suavizamiento exponencial doble)",
            family="suavizamiento", seasonal=False, constant_level=False,
            _forecast=_f_holt, default_params={"alpha": 0.2, "beta": 0.1},
            _grid=_grid_holt, _min_obs=lambda m: 6,
        ),
        ModelSpec(
            key="holt_winters", label="Holt-Winters (aditivo)", family="suavizamiento",
            seasonal=True, constant_level=False, _forecast=_f_holt_winters,
            default_params={"alpha": 0.2, "beta": 0.1, "gamma": 0.1},
            _grid=_grid_holt_winters, _min_obs=lambda m: 2 * m,
        ),
        ModelSpec(
            key="linear_regression", label="Regresion lineal sobre el tiempo",
            family="regresion", seasonal=False, constant_level=False,
            _forecast=_f_linear_regression, _min_obs=lambda m: 4,
        ),
    ]
    if STATSFORECAST_AVAILABLE:
        specs += [
            ModelSpec(
                key="auto_arima", label="ARIMA (orden por AICc)", family="arima",
                seasonal=False, constant_level=False, _forecast=_f_auto_arima,
                _min_obs=lambda m: 10, analytic_interval=True,
                note="Sustituye el barrido manual de (p,d,q): seleccion por AICc, "
                     "no por MAPE post-hoc (F05, F13).",
            ),
            ModelSpec(
                key="auto_sarima", label="SARIMA (orden por AICc)", family="arima",
                seasonal=True, constant_level=False, _forecast=_f_auto_sarima,
                _min_obs=lambda m: max(10, 2 * m), analytic_interval=True,
                note="Sustituye el barrido de 144 combinaciones (p,d,q)(P,D,Q).",
            ),
            ModelSpec(
                key="auto_ets", label="ETS (familia por AICc)", family="suavizamiento",
                seasonal=True, constant_level=False, _forecast=_f_auto_ets,
                _min_obs=lambda m: max(8, 2 * m if m > 1 else 8), analytic_interval=True,
                note="Comparador moderno equivalente a ets() de R.",
            ),
            ModelSpec(
                key="auto_theta", label="Theta (variante por AICc)", family="theta",
                seasonal=True, constant_level=False, _forecast=_f_auto_theta,
                _min_obs=lambda m: max(8, 2 * m if m > 1 else 8), analytic_interval=True,
                note="Ganador de la M3; comparador de referencia en series cortas.",
            ),
        ]
    return {s.key: s for s in specs}


MODEL_REGISTRY: dict[str, ModelSpec] = _build_registry()


def available_keys() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def get_spec(key: str) -> ModelSpec:
    """Despacho por clave EXACTA. Una clave desconocida es un error (F04)."""
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            "Modelo desconocido: {!r}. Disponibles: {}".format(
                key, ", ".join(sorted(MODEL_REGISTRY))
            )
        ) from None


def forecast_with(key: str, train, params=None, h=1, m=12, clip_non_negative=True):
    return get_spec(key).forecast(
        train, params=params, h=h, m=m, clip_non_negative=clip_non_negative
    )


def eligible_specs(
    profile,
    *,
    n_train: int,
    structural_filter: bool = True,
    keys: list[str] | None = None,
) -> tuple[list[ModelSpec], dict[str, str]]:
    """Selecciona los modelos candidatos segun la estructura de la serie (F22a).

    Implementa el filtrado por caracteristicas que el manuscrito afirmaba tener
    (Montero-Manso et al. 2020; Talagala et al. 2021) y que no existia en el
    codigo. Cada exclusion queda registrada con su motivo: nada se descarta en
    silencio.

    Devuelve (candidatos, {clave: motivo_de_exclusion}).
    """
    from .classification import allow_constant_level_methods

    pool = [MODEL_REGISTRY[k] for k in (keys or MODEL_REGISTRY)]
    m_eff = profile.seasonal_period
    allow_flat = allow_constant_level_methods(profile)

    chosen: list[ModelSpec] = []
    excluded: dict[str, str] = {}
    for spec in pool:
        need = spec.min_obs(profile.m if spec.seasonal else m_eff)
        if n_train < need:
            excluded[spec.key] = (
                "historia insuficiente: requiere {} observaciones de "
                "entrenamiento, hay {}".format(need, n_train)
            )
            continue
        if not structural_filter:
            chosen.append(spec)
            continue
        if spec.seasonal and spec.key != "seasonal_naive" and not profile.has_seasonality:
            excluded[spec.key] = (
                "sin estacionalidad detectada (F_S={:.3f} < {:.2f})".format(
                    profile.seasonal_strength, 0.40
                )
            )
            continue
        if spec.constant_level and spec.key != "naive" and not allow_flat:
            motivo = []
            if profile.has_trend:
                motivo.append("tendencia significativa (p={:.4f})".format(profile.trend_pvalue))
            if not profile.is_stationary:
                motivo.append("nivel no estacionario ({})".format(profile.stationarity_verdict))
            excluded[spec.key] = (
                "pronostico de nivel constante incompatible con un nivel "
                "inestable: " + "; ".join(motivo or ["nivel no estable"])
            )
            continue
        chosen.append(spec)

    # Los benchmarks entran siempre que la historia lo permita: sin ellos no hay
    # forma de interpretar ninguna metrica (F12).
    for key in ("naive", "seasonal_naive"):
        spec = MODEL_REGISTRY[key]
        if spec in chosen:
            continue
        if n_train >= spec.min_obs(profile.m):
            chosen.append(spec)
            excluded.pop(key, None)

    order = {k: i for i, k in enumerate(MODEL_REGISTRY)}
    chosen.sort(key=lambda s: order[s.key])
    return chosen, excluded
