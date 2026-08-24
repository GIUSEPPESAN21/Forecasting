"""Seleccion de hiperparametros y del modelo ganador, sin sesgo de seleccion.

Resuelve F05 y F13.

El problema original
--------------------
El pipeline (1) evaluaba 9 metodos por walk-forward sobre TODA la serie,
(2) elegia el de menor MAPE, (3) barria hasta 361 combinaciones minimizando
*ese mismo* MAPE y (4) reportaba el minimo resultante como desempeno. Con 18
puntos de evaluacion y 361 candidatos, el minimo observado esta dominado por
ruido de seleccion: es un limite inferior optimista, no una estimacion del
error futuro.

El protocolo de aqui
--------------------
Los origenes del walk-forward se parten en dos bloques disjuntos y contiguos:

    |-------- tune --------||---- eval ----|
     hiperparametros aqui     metrica reportada aqui

Ningun hiperparametro ve el bloque `eval`, y la metrica publicada nunca es el
minimo de un barrido. Ademas, para ARIMA/SARIMA el barrido desaparece por
completo: `AutoARIMA` selecciona el orden por AICc (verosimilitud penalizada
sobre el train), que no consume el bloque de evaluacion y es a la vez mas
rapido y mas defendible (F13).

`honest_outer_estimate` va un paso mas alla y repite TODO el protocolo
—incluida la eleccion del metodo— sobre una serie truncada, para estimar el
desempeno sin el sesgo de haber elegido tambien el metodo sobre el bloque de
evaluacion. Es el numero que debe ir al manuscrito.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .classification import SeriesProfile, classify_series
from .models import MODEL_REGISTRY, ModelSpec, eligible_specs
from .validation import (
    ABSOLUTE_MIN_TRAIN,
    MIN_ORIGINS,
    WalkForwardResult,
    resolve_min_train,
    rolling_origins,
    walk_forward,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TuningResult", "PipelineResult", "tune_spec", "run_pipeline",
    "honest_outer_estimate", "DEFAULT_EVAL_BLOCK", "TUNE_ORIGIN_CAP",
]

# Origenes reservados para medir (nunca para elegir hiperparametros).
DEFAULT_EVAL_BLOCK = 10
# Tope de origenes usados para BUSCAR hiperparametros: acota el costo a
# O(candidatos x TUNE_ORIGIN_CAP) sin importar el largo de la serie.
TUNE_ORIGIN_CAP = 12
# Origenes minimos en el bloque de tuning para que la busqueda tenga sentido.
MIN_TUNE_ORIGINS = 4


def _numeric_scalar_keys(params: dict) -> list[str]:
    return [
        k for k, v in params.items()
        if isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        and 0.0 < float(v) < 1.0
    ]


def refine_around(best: dict, bounds: tuple[float, float] = (0.02, 0.98)) -> list[dict]:
    """Busqueda local fina alrededor del mejor punto grueso (coarse-to-fine).

    Con 1-2 parametros usa una rejilla de 5 puntos por eje; con 3 baja a 3
    puntos para que el producto cartesiano no vuelva a explotar (F13).
    """
    keys = _numeric_scalar_keys(best)
    if not keys:
        return []
    deltas = (-0.10, -0.05, 0.0, 0.05, 0.10) if len(keys) <= 2 else (-0.075, 0.0, 0.075)
    lo, hi = bounds
    axes = []
    for k in keys:
        base = float(best[k])
        axes.append(sorted({round(min(max(base + d, lo), hi), 4) for d in deltas}))
    out = []
    for combo in product(*axes):
        cand = dict(best)
        cand.update(dict(zip(keys, combo)))
        out.append(cand)
    seen, uniq = set(), []
    for c in out:
        sig = tuple(sorted((k, str(v)) for k, v in c.items()))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


@dataclass
class TuningResult:
    key: str
    params: dict
    tuned: bool
    reason: str
    n_evaluated: int
    tune_mase: float
    seconds: float
    trace: list[tuple[dict, float]] = field(default_factory=list)


def _score_on(y, spec, params, m, season_length, origins, scale_train) -> float:
    """MASE de un candidato sobre un conjunto de origenes dado."""
    res = walk_forward(
        y, [spec], m=m, season_length=season_length, min_train=int(origins[0]),
        params_by_key={spec.key: params}, origins=origins, scale_train=scale_train,
    )
    row = res.metrics.iloc[0]
    if not bool(row["elegible"]):
        return float("inf")
    val = float(row["mase"])
    return val if np.isfinite(val) else float("inf")


def tune_spec(
    y,
    spec: ModelSpec,
    *,
    m: int,
    season_length: int,
    tune_origins: np.ndarray,
    scale_train: np.ndarray,
    n_obs: int,
    refine: bool = True,
) -> TuningResult:
    """Elige hiperparametros usando SOLO el bloque de tuning."""
    t0 = time.perf_counter()
    if not spec.has_hyperparameters:
        return TuningResult(
            spec.key, dict(spec.default_params), False,
            "el modelo no tiene hiperparametros que ajustar", 0, float("nan"),
            time.perf_counter() - t0,
        )
    if tune_origins.size < MIN_TUNE_ORIGINS:
        return TuningResult(
            spec.key, dict(spec.default_params), False,
            "bloque de tuning insuficiente ({} origenes, minimo {}): se usan los "
            "parametros por defecto y se declara asi".format(
                tune_origins.size, MIN_TUNE_ORIGINS
            ),
            0, float("nan"), time.perf_counter() - t0,
        )

    origins = tune_origins[-TUNE_ORIGIN_CAP:]
    trace: list[tuple[dict, float]] = []

    best_params, best_score = None, float("inf")
    for cand in spec.grid(n_obs, m):
        score = _score_on(y, spec, cand, m, season_length, origins, scale_train)
        trace.append((dict(cand), score))
        if score < best_score:
            best_params, best_score = dict(cand), score

    if best_params is None:
        return TuningResult(
            spec.key, dict(spec.default_params), False,
            "ningun candidato de la parrilla pudo evaluarse", len(trace),
            float("nan"), time.perf_counter() - t0, trace,
        )

    if refine:
        for cand in refine_around(best_params):
            if any(cand == c for c, _ in trace):
                continue
            score = _score_on(y, spec, cand, m, season_length, origins, scale_train)
            trace.append((dict(cand), score))
            if score < best_score:
                best_params, best_score = dict(cand), score

    return TuningResult(
        spec.key, best_params, True,
        "ajustado sobre {} origenes del bloque de tuning; el bloque de "
        "evaluacion no participo".format(origins.size),
        len(trace), float(best_score), time.perf_counter() - t0, trace,
    )


@dataclass
class PipelineResult:
    """Todo lo necesario para explicar como se llego al ganador."""

    profile: SeriesProfile
    m: int
    min_train: int
    origins: np.ndarray
    tune_origins: np.ndarray
    eval_origins: np.ndarray
    candidates: list[str]
    excluded: dict[str, str]
    tuning: dict[str, TuningResult]
    evaluation: WalkForwardResult
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors and not self.evaluation.metrics.empty

    @property
    def winner(self) -> str | None:
        return self.evaluation.winner

    @property
    def winner_params(self) -> dict:
        w = self.winner
        if w is None:
            return {}
        return dict(self.tuning[w].params) if w in self.tuning else {}

    def ranking(self) -> pd.DataFrame:
        cols = ["modelo", "etiqueta", "mase", "mape", "mad", "rmse", "me",
                "tracking_signal", "n_preds", "params", "elegible", "motivo"]
        df = self.evaluation.metrics
        return df[[c for c in cols if c in df.columns]]


def run_pipeline(
    series,
    *,
    m: int = 12,
    eval_block: int = DEFAULT_EVAL_BLOCK,
    structural_filter: bool = True,
    keys: list[str] | None = None,
    profile: SeriesProfile | None = None,
    clip_non_negative: bool = True,
) -> PipelineResult:
    """Clasifica, filtra, ajusta y evalua. Metrica reportada = bloque `eval`."""
    t0 = time.perf_counter()
    s = pd.Series(series).astype(float)
    dates = s.index if isinstance(s.index, pd.DatetimeIndex) else None
    y = s.to_numpy(dtype=float)
    n = y.size
    notes: list[str] = []
    errors: list[str] = []

    prof = profile or classify_series(s.dropna(), m=m)
    notes.extend(prof.warnings)

    include_seasonal = prof.has_seasonality
    min_train = resolve_min_train(n, m, include_seasonal=include_seasonal)
    origins = rolling_origins(n, min_train)

    if origins.size < MIN_ORIGINS and include_seasonal:
        # No alcanza para validar modelos estacionales: se declara y se reduce el
        # conjunto de candidatos, en vez de bajar min_train en silencio (F02).
        notes.append(
            "Se detecto estacionalidad (F_S={:.3f}) pero con {} observaciones solo "
            "quedan {} origenes de validacion con min_train=2*m={}. Los modelos "
            "estacionales NO pueden validarse honestamente y quedan fuera del "
            "ranking; se evalua el conjunto no estacional con min_train={}.".format(
                prof.seasonal_strength, n, origins.size, min_train, resolve_min_train(
                    n, m, include_seasonal=False)
            )
        )
        include_seasonal = False
        min_train = resolve_min_train(n, m, include_seasonal=False)
        origins = rolling_origins(n, min_train)

    if origins.size < MIN_ORIGINS:
        errors.append(
            "Solo {} origenes de validacion (minimo {}). Se requieren al menos {} "
            "observaciones para evaluar cualquier metodo con honestidad.".format(
                origins.size, MIN_ORIGINS, min_train + MIN_ORIGINS
            )
        )
        empty = walk_forward(y, [], m=m, min_train=min_train, origins=np.array([], int))
        return PipelineResult(
            prof, m, min_train, origins, np.array([], int), np.array([], int),
            [], {}, {}, empty, notes, errors, time.perf_counter() - t0,
        )

    m_eff = m if include_seasonal else 1
    specs, excluded = eligible_specs(
        prof, n_train=int(origins[0]), structural_filter=structural_filter, keys=keys
    )
    if not include_seasonal:
        for spec in list(specs):
            if spec.seasonal and spec.key != "seasonal_naive":
                specs.remove(spec)
                excluded[spec.key] = (
                    "modelo estacional sin historia suficiente para validarse "
                    "({} origenes disponibles)".format(origins.size)
                )

    # --- particion tune / eval (F05) -------------------------------------
    k_eval = int(min(max(eval_block, MIN_ORIGINS // 2), origins.size))
    if origins.size - k_eval < MIN_TUNE_ORIGINS:
        k_eval = max(MIN_ORIGINS // 2, origins.size - MIN_TUNE_ORIGINS)
        k_eval = max(1, min(k_eval, origins.size))
    tune_origins = origins[:-k_eval] if k_eval < origins.size else np.array([], int)
    eval_origins = origins[-k_eval:]

    if tune_origins.size < MIN_TUNE_ORIGINS:
        notes.append(
            "Solo {} origenes disponibles para ajustar hiperparametros (minimo {}): "
            "todos los modelos usan sus parametros por defecto y asi se reporta. "
            "Ningun numero publicado proviene de un minimo de barrido.".format(
                tune_origins.size, MIN_TUNE_ORIGINS
            )
        )

    scale_train = y[: origins[0]]
    tuning: dict[str, TuningResult] = {}
    for spec in specs:
        tuning[spec.key] = tune_spec(
            y, spec, m=m_eff, season_length=m,
            tune_origins=tune_origins, scale_train=scale_train, n_obs=n,
        )

    # --- evaluacion final: bloque intacto --------------------------------
    eval_scale = y[: eval_origins[0]]
    evaluation = walk_forward(
        y, specs, m=m_eff, season_length=m, min_train=int(eval_origins[0]),
        params_by_key={k: t.params for k, t in tuning.items()},
        dates=dates, origins=eval_origins, scale_train=eval_scale,
        clip_non_negative=clip_non_negative,
    )
    notes.append(
        "Metrica reportada sobre {} origenes ({} a {}) que no participaron en la "
        "eleccion de hiperparametros.".format(
            eval_origins.size,
            "{:%Y-%m}".format(dates[eval_origins[0]]) if dates is not None else eval_origins[0],
            "{:%Y-%m}".format(dates[eval_origins[-1]]) if dates is not None else eval_origins[-1],
        )
    )
    if evaluation.disqualified:
        notes.append(
            "Descalificados por no cubrir todos los origenes: {}.".format(
                ", ".join(sorted(evaluation.disqualified))
            )
        )

    return PipelineResult(
        profile=prof, m=m, min_train=min_train, origins=origins,
        tune_origins=tune_origins, eval_origins=eval_origins,
        candidates=[s.key for s in specs], excluded=excluded, tuning=tuning,
        evaluation=evaluation, notes=notes, errors=errors,
        seconds=time.perf_counter() - t0,
    )


def honest_outer_estimate(
    series,
    *,
    m: int = 12,
    outer_block: int = 6,
    eval_block: int = DEFAULT_EVAL_BLOCK,
    structural_filter: bool = True,
    keys: list[str] | None = None,
) -> dict:
    """Desempeno sin sesgo de seleccion: el metodo TAMBIEN se elige a ciegas.

    `run_pipeline` deja el bloque `eval` fuera de la eleccion de
    hiperparametros, pero el metodo ganador si se elige mirando ese bloque. Para
    el numero que va al manuscrito hay que ir un paso mas: se reserva un bloque
    exterior, se repite todo el protocolo sobre la serie truncada, y se evalua
    el ganador resultante sobre el bloque exterior, que no vio nada.
    """
    s = pd.Series(series).astype(float).dropna()
    n = s.size
    # El pipeline interno debe ser el mismo protocolo completo, con bloque de
    # tuning propio. Sin eso la "estimacion honesta" sale de una corrida
    # degenerada y no vale mas que la interna.
    minimo_interno = ABSOLUTE_MIN_TRAIN + MIN_TUNE_ORIGINS + MIN_ORIGINS
    if n - outer_block < minimo_interno:
        return {
            "ok": False,
            "reason": "serie demasiado corta para una estimacion externa: {} "
                      "observaciones menos un bloque externo de {} dejan {} para el "
                      "pipeline interno, que necesita al menos {} "
                      "(min_train {} + tuning {} + evaluacion {})".format(
                          n, outer_block, n - outer_block, minimo_interno,
                          ABSOLUTE_MIN_TRAIN, MIN_TUNE_ORIGINS, MIN_ORIGINS),
        }
    inner = s.iloc[: n - outer_block]
    inner_res = run_pipeline(
        inner, m=m, eval_block=eval_block, structural_filter=structural_filter, keys=keys
    )
    if not inner_res.ok or inner_res.winner is None:
        return {"ok": False, "reason": "el pipeline interno no produjo ganador",
                "inner": inner_res}

    winner = inner_res.winner
    spec = MODEL_REGISTRY[winner]
    params = inner_res.winner_params
    m_eff = m if inner_res.profile.has_seasonality else 1

    outer_origins = np.arange(n - outer_block, n, dtype=int)
    # El ganador puede SER naive o seasonal_naive: no duplicar la clave, o
    # walk_forward devuelve dos filas "naive" y cualquier .loc["naive"]
    # aguas abajo se vuelve ambiguo.
    outer_specs = {spec.key: spec}
    outer_specs.setdefault("naive", MODEL_REGISTRY["naive"])
    outer_specs.setdefault("seasonal_naive", MODEL_REGISTRY["seasonal_naive"])
    outer = walk_forward(
        s, list(outer_specs.values()),
        m=m_eff, season_length=m, min_train=int(outer_origins[0]),
        params_by_key={winner: params}, dates=s.index, origins=outer_origins,
        scale_train=s.to_numpy()[: outer_origins[0]],
    )
    return {
        "ok": True,
        "winner": winner,
        "params": params,
        "inner": inner_res,
        "outer": outer,
        "outer_metrics": outer.metrics,
        "n_outer": int(outer_block),
    }
