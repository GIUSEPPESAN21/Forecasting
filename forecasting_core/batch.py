"""Procesamiento multi-SKU con memoria acotada (F19).

La herramienta original procesaba una serie por carga (`multiple=False`). Un
planeador con 200 referencias tenia que subir 200 archivos y esperar entre 30 s
y varios minutos por cada uno. La escalabilidad que el manuscrito reclamaba
demostrar era en LONGITUD de serie, no en NUMERO de series, que es la dimension
que realmente limita la adopcion.

Presupuesto de memoria
----------------------
El objetivo es un PC de 8 GB (~5-6 GB utiles). Dos reglas lo hacen posible:

1. **Nunca acumular el portafolio en memoria.** Cada SKU se procesa, se
   serializa su fila de resultado y se descarta. Las filas se vuelcan a disco
   en bloques (`flush_every`), de modo que el pico de memoria depende del
   tamano del bloque, no del numero de SKU.
2. **Paralelismo acotado.** Cada worker que importa statsmodels/statsforecast
   cuesta 150-300 MB solo en el import, asi que `n_jobs` nunca es -1:
   `min(4, cpu_count - 2)`, y con un tope adicional por RAM disponible.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .classification import classify_series
from .data import LoadResult, load_panel
from .intervals import prediction_interval
from .inventory import compute_policy
from .models import get_spec
from .optimize import run_pipeline

logger = logging.getLogger(__name__)

__all__ = ["BatchConfig", "BatchSummary", "resolve_n_jobs", "process_one", "run_batch"]

# Memoria estimada por worker (import de statsmodels + statsforecast + datos).
WORKER_MB = 320
# Tope duro de workers, independiente del numero de nucleos.
MAX_WORKERS = 4


def resolve_n_jobs(requested: int | None = None, available_mb: int | None = None) -> int:
    """n_jobs = min(4, nucleos-2), acotado ademas por la RAM disponible.

    Nunca devuelve -1 ni el total de nucleos: en un equipo de 8 GB, lanzar 12
    procesos que importan statsmodels agota la memoria antes de terminar.
    """
    cpus = os.cpu_count() or 2
    by_cpu = max(1, min(MAX_WORKERS, cpus - 2))
    if requested is not None and requested > 0:
        by_cpu = min(by_cpu, int(requested))
    if available_mb:
        by_cpu = max(1, min(by_cpu, int(available_mb // WORKER_MB)))
    return by_cpu


@dataclass
class BatchConfig:
    m: int = 12
    horizon: int = 12
    level: float = 0.95
    lead_time: int = 3
    service_level: float = 0.95
    eval_block: int = 10
    structural_filter: bool = True
    gap_policy: str = "report"
    n_jobs: int | None = None
    flush_every: int = 25
    with_intervals: bool = True
    with_inventory: bool = True


@dataclass
class BatchSummary:
    n_skus: int = 0
    n_ok: int = 0
    n_failed: int = 0
    seconds: float = 0.0
    output_path: Path | None = None
    forecast_path: Path | None = None
    peak_rows_in_memory: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            "{}/{} SKU procesados en {:.1f} s ({:.2f} s/SKU) | fallidos: {}".format(
                self.n_ok, self.n_skus, self.seconds,
                self.seconds / max(1, self.n_skus), self.n_failed,
            )
        )


def process_one(sku: str, load: LoadResult, cfg: BatchConfig) -> tuple[dict, pd.DataFrame | None]:
    """Procesa un SKU y devuelve (fila_resumen, tabla_de_pronostico)."""
    t0 = time.perf_counter()
    row: dict = {"sku": sku, "n_obs": load.report.n_obs, "estado": "ok", "detalle": ""}

    if not load.report.ok:
        row.update(estado="error_carga", detalle=" | ".join(load.report.errors))
        return row, None

    s = load.series.dropna()
    try:
        res = run_pipeline(
            s, m=cfg.m, eval_block=cfg.eval_block,
            structural_filter=cfg.structural_filter,
        )
    except Exception as exc:
        logger.exception("SKU %s: el pipeline fallo", sku)
        row.update(estado="error_pipeline", detalle="{}: {}".format(type(exc).__name__, exc))
        return row, None

    prof = res.profile
    row.update(
        inicio="{:%Y-%m}".format(s.index.min()), fin="{:%Y-%m}".format(s.index.max()),
        tendencia=prof.has_trend, p_tendencia=round(prof.trend_pvalue, 4),
        estacionalidad=prof.has_seasonality,
        fuerza_estacional=(None if not np.isfinite(prof.seasonal_strength)
                           else round(prof.seasonal_strength, 3)),
        estacionariedad=prof.stationarity_verdict,
        baja_potencia=prof.low_power,
        n_origenes=int(res.origins.size), n_eval=int(res.eval_origins.size),
    )

    if not res.ok or res.winner is None:
        row.update(estado="sin_ganador", detalle=" | ".join(res.errors) or "sin metodos elegibles")
        row["segundos"] = round(time.perf_counter() - t0, 2)
        return row, None

    best = res.evaluation.ranked.iloc[0]
    naive_row = res.evaluation.metrics.query("modelo == 'naive'")
    row.update(
        metodo=res.winner, params=str(res.winner_params),
        mase=round(float(best["mase"]), 4), mape=round(float(best["mape"]), 2),
        mad=round(float(best["mad"]), 2), me=round(float(best["me"]), 2),
        senal_rastreo=round(float(best["tracking_signal"]), 2),
        mase_naive=(round(float(naive_row.iloc[0]["mase"]), 4) if len(naive_row) else None),
        supera_naive=(bool(float(best["mase"]) < float(naive_row.iloc[0]["mase"]))
                      if len(naive_row) else None),
    )

    fc_table = None
    if cfg.with_intervals:
        try:
            spec = get_spec(res.winner)
            pi = prediction_interval(
                s, spec, res.winner_params, season_length=cfg.m,
                horizon=cfg.horizon, level=cfg.level,
            )
            fc_table = pi.to_frame().reset_index(names="fecha")
            fc_table.insert(0, "sku", sku)
            fc_table.insert(1, "metodo", res.winner)
            row["metodo_intervalo"] = pi.method
        except Exception as exc:
            logger.warning("SKU %s: intervalos fallaron (%s)", sku, exc)
            row["detalle"] = (row["detalle"] + " | intervalos: {}".format(exc)).strip(" |")

    if cfg.with_inventory:
        try:
            pol = compute_policy(
                s, get_spec(res.winner), res.winner_params,
                lead_time=cfg.lead_time, service_level=cfg.service_level,
                season_length=cfg.m,
            )
            row.update(
                demanda_lead_time=round(pol.demand_lead_time, 1),
                sigma_lead_time=round(pol.sigma_lead_time, 1),
                stock_seguridad=round(pol.safety_stock, 1),
                punto_reorden=round(pol.reorder_point, 1),
                avisos_inventario=" | ".join(pol.warnings),
            )
        except Exception as exc:
            logger.warning("SKU %s: politica de inventario fallo (%s)", sku, exc)

    row["segundos"] = round(time.perf_counter() - t0, 2)
    return row, fc_table


def _flush(rows: list[dict], path: Path, header_written: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a" if header_written else "w",
              header=not header_written, index=False, encoding="utf-8-sig")


def run_batch(
    panel: pd.DataFrame,
    output_dir: str | Path,
    cfg: BatchConfig | None = None,
    *,
    progress=None,
) -> BatchSummary:
    """Procesa un panel multi-SKU volcando resultados a disco incrementalmente.

    El pico de memoria depende de `cfg.flush_every`, no del numero de SKU: se
    verifica en `tests/test_batch_memory.py`.
    """
    cfg = cfg or BatchConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "resumen_skus.csv"
    forecast_path = out / "pronosticos.csv"
    for p in (summary_path, forecast_path):
        if p.exists():
            p.unlink()

    loads = load_panel(panel, gap_policy=cfg.gap_policy)
    summary = BatchSummary(n_skus=len(loads), output_path=summary_path,
                           forecast_path=forecast_path)
    t0 = time.perf_counter()

    rows: list[dict] = []
    header_summary = False
    header_forecast = False

    for i, (sku, load) in enumerate(loads.items(), start=1):
        try:
            row, fc = process_one(sku, load, cfg)
        except Exception as exc:  # red de seguridad: un SKU nunca tumba el lote
            logger.exception("SKU %s: fallo no controlado", sku)
            row, fc = {"sku": sku, "estado": "error", "detalle": str(exc)}, None

        rows.append(row)
        if row.get("estado") == "ok":
            summary.n_ok += 1
        else:
            summary.n_failed += 1
            summary.failures[sku] = str(row.get("detalle", ""))[:300]

        if fc is not None and len(fc):
            fc.to_csv(forecast_path, mode="a" if header_forecast else "w",
                      header=not header_forecast, index=False, encoding="utf-8-sig")
            header_forecast = True

        summary.peak_rows_in_memory = max(summary.peak_rows_in_memory, len(rows))
        if len(rows) >= cfg.flush_every:
            _flush(rows, summary_path, header_summary)
            header_summary = True
            rows.clear()
            gc.collect()

        if progress is not None:
            progress(i, len(loads), sku)

    _flush(rows, summary_path, header_summary)
    summary.seconds = time.perf_counter() - t0
    return summary
