"""Medicion rigurosa de tiempos de computo (Fase 7 / F25).

La Tabla 3 original se midio con cronometro manual, n=1 corrida, y NO incluia
el Modulo 3 (optimizacion de hiperparametros) — que la Auditoria B midio como
~60% del tiempo total (183.05 s reales vs. 89.79 s reportados, n=120).

Este script corrige los tres defectos a la vez:
  1. `time.perf_counter()` con >= REPS repeticiones por tamano de serie.
  2. Reporta media +/- desviacion estandar, no un solo numero.
  3. Incluye el pipeline COMPLETO (clasificacion + evaluacion + tuning +
     pronostico + intervalo), que es lo que el usuario realmente espera.

Uso:
    python experiments/benchmark_tiempos.py --reps 10 --sizes 24 48 72 96 120
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import scipy
import statsmodels
import statsforecast

from forecasting_core.intervals import prediction_interval  # noqa: E402
from forecasting_core.models import get_spec  # noqa: E402
from forecasting_core.optimize import run_pipeline  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"


def make_series(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    y = 1000 + 12 * t + 250 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 90, n)
    return pd.Series(y, index=idx)


def time_full_pipeline(s: pd.Series) -> float:
    """Pipeline completo tal como lo ejecuta un usuario: clasificar, evaluar,
    afinar, pronosticar con intervalo. Es el numero comparable con la Tabla 3
    original SI se le suma el Modulo 3 (aqui va incluido de entrada)."""
    t0 = time.perf_counter()
    res = run_pipeline(s, m=12)
    if res.ok and res.winner:
        spec = get_spec(res.winner)
        prediction_interval(s, spec, res.winner_params, season_length=12, horizon=12)
    return time.perf_counter() - t0


def fit_complexity(sizes: np.ndarray, seconds: np.ndarray) -> tuple[float, float]:
    """Ajusta seconds ~ a * n^b (log-log) y devuelve (a, b)."""
    mask = seconds > 0
    if mask.sum() < 2:
        return float("nan"), float("nan")
    log_n = np.log(sizes[mask])
    log_t = np.log(seconds[mask])
    b, log_a = np.polyfit(log_n, log_t, 1)
    return float(np.exp(log_a)), float(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 72, 96, 120])
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    print("Entorno:")
    print("  Python      :", sys.version.split()[0])
    print("  Plataforma  :", platform.platform())
    print("  CPU logicos :", __import__("os").cpu_count())
    print("  pandas {} | numpy {} | scipy {} | statsmodels {} | statsforecast {}".format(
        pd.__version__, np.__version__, scipy.__version__,
        statsmodels.__version__, statsforecast.__version__))
    print()

    rows = []
    for n in args.sizes:
        times = []
        for r in range(args.reps):
            s = make_series(n, seed=args.seed + r)
            times.append(time_full_pipeline(s))
        arr = np.array(times)
        rows.append({
            "n": n, "reps": args.reps,
            "media_s": arr.mean(), "sd_s": arr.std(ddof=1) if arr.size > 1 else 0.0,
            "min_s": arr.min(), "max_s": arr.max(),
        })
        print("n={:4d}  media={:6.2f}s  sd={:5.2f}s  min={:6.2f}s  max={:6.2f}s  "
              "({} repeticiones)".format(n, arr.mean(), arr.std(ddof=1) if arr.size > 1 else 0.0,
                                         arr.min(), arr.max(), args.reps))

    df = pd.DataFrame(rows)
    a, b = fit_complexity(df["n"].to_numpy(dtype=float), df["media_s"].to_numpy())

    print("\n" + "=" * 78)
    print("Ajuste de complejidad empirica: tiempo ~ {:.2e} * n^{:.2f}".format(a, b))
    if b >= 2.5:
        etiqueta = "cubica o peor"
    elif b >= 1.7:
        etiqueta = "aproximadamente cuadratica"
    elif b >= 1.2:
        etiqueta = "superlineal"
    else:
        etiqueta = "aproximadamente lineal"
    print("  Interpretacion: crecimiento {} (exponente {:.2f}).".format(etiqueta, b))
    print("  [Auditoria A midio un exponente ~2.0 sobre los datos del propio manuscrito;")
    print("   el original lo describia como 'increases progressively', sin caracterizarlo.]")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "benchmark_tiempos.csv"
    df.to_csv(path, index=False)
    print("\nCSV:", path)
    print("\nPresupuesto de la Fase 1 del prompt maestro (sesion interactiva, n<=48):")
    print("  objetivo: pipeline completo <= 25 s")
    for _, row in df[df["n"] <= 48].iterrows():
        estado = "OK" if row["media_s"] <= 25 else "EXCEDE"
        print("  n={:3d}: media {:.2f}s -> {}".format(int(row["n"]), row["media_s"], estado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
