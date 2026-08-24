"""CLI para procesamiento por lotes multi-SKU (F19).

Un planeador con 200 referencias no debería tener que escribir Python para
usar `forecasting_core.batch`. Este script es la puerta de entrada de linea
de comandos.

Uso:
    python batch_cli.py productos.xlsx salida/ --horizon 12 --lead-time 3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from forecasting_core.batch import BatchConfig, resolve_n_jobs, run_batch

logging.basicConfig(
    level=os.environ.get("FORECASTING_LOG_LEVEL", "WARNING"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Excel/CSV con columnas sku, year, month, demand")
    ap.add_argument("output_dir", help="Directorio de salida (resumen_skus.csv, pronosticos.csv)")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--level", type=float, default=0.95, help="Nivel del intervalo de prediccion")
    ap.add_argument("--lead-time", type=int, default=3)
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--gap-policy", choices=["report", "interpolate", "zero", "abort"], default="report")
    ap.add_argument("--no-intervals", action="store_true")
    ap.add_argument("--no-inventory", action="store_true")
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()

    if args.input.lower().endswith(".csv"):
        panel = pd.read_csv(args.input, dtype=str)
    else:
        panel = pd.read_excel(args.input, dtype=str)

    cfg = BatchConfig(
        horizon=args.horizon, level=args.level, lead_time=args.lead_time,
        service_level=args.service_level, gap_policy=args.gap_policy,
        with_intervals=not args.no_intervals, with_inventory=not args.no_inventory,
        flush_every=args.flush_every,
    )

    def progress(i, n, sku):
        if i % 10 == 0 or i == n:
            print("[{}/{}] {}".format(i, n, sku), file=sys.stderr)

    print("n_jobs recomendado para este equipo: {} (nunca -1; ver Fase 1 del "
          "refactor sobre presupuesto de memoria)".format(resolve_n_jobs()))
    summary = run_batch(panel, args.output_dir, cfg, progress=progress)
    print(summary.describe())
    if summary.failures:
        print("\nSKU con problemas:")
        for sku, motivo in list(summary.failures.items())[:20]:
            print("  {}: {}".format(sku, motivo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
