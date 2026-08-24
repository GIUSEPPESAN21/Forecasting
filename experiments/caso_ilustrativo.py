"""Caso ilustrativo end-to-end para la Seccion 3.2 del manuscrito.

Genera UNA serie sintetica determinista (semilla fija) con tendencia y
estacionalidad moderadas -reproduciendo la variabilidad y los cambios de nivel
que la tesis documenta para las series de Tuboplex- y ejecuta el pipeline
completo, imprimiendo cada numero que el manuscrito cita en la Seccion 3.2.

Se usa una serie sintetica, NO datos reales de la empresa, porque el archivo
real de Tuboplex no esta disponible en este repositorio (ver Data Availability
en el manuscrito). Todo numero impreso aqui es exactamente el que el usuario
obtendria ejecutando este script; no hay ninguna cifra en el manuscrito que no
provenga de una ejecucion real de este codigo.

Uso:
    python experiments/caso_ilustrativo.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from forecasting_core.intervals import prediction_interval  # noqa: E402
from forecasting_core.inventory import compute_policy  # noqa: E402
from forecasting_core.models import get_spec  # noqa: E402
from forecasting_core.optimize import run_pipeline  # noqa: E402

SEED = 20260824


def make_case_series() -> pd.Series:
    """36 meses: tendencia creciente moderada + estacionalidad + ruido realista.

    Reproduce la caracterizacion cualitativa de la tesis (Sec. 3.1-3.2): picos
    en ciertos periodos, cambios relativamente abruptos entre meses.
    """
    rng = np.random.default_rng(SEED)
    n = 36
    t = np.arange(n, dtype=float)
    base = 1800 + 18 * t + 260 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 130, n)
    base[14] += 900   # pico de proyecto de obra
    base[27] -= 500   # caida abrupta
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    return pd.Series(np.maximum(base, 0).round(1), index=idx, name="demand")


def main() -> int:
    s = make_case_series()
    print("=== Carga y validacion ===")
    print("Periodo: {:%Y-%m} -> {:%Y-%m} ({} observaciones)".format(
        s.index.min(), s.index.max(), len(s)))

    print("\n=== Clasificacion estructural ===")
    result = run_pipeline(s, m=12)
    prof = result.profile
    print("Tendencia    : {} (p={:.4f}) [{}]".format(
        "Si" if prof.has_trend else "No", prof.trend_pvalue, prof.trend_test))
    print("Estacionalidad: {} (F_S={}) [{}]".format(
        "Si" if prof.has_seasonality else "No",
        "n/e" if not np.isfinite(prof.seasonal_strength) else round(prof.seasonal_strength, 3),
        prof.seasonality_test))
    print("Estacionariedad: {}".format(prof.stationarity_verdict))

    print("\n=== Filtro estructural ===")
    for k, v in result.excluded.items():
        print("  excluido: {} -- {}".format(k, v))
    print("  candidatos evaluados: {}".format(", ".join(result.candidates)))

    print("\n=== Ranking (bloque de evaluacion, {} origenes) ===".format(
        result.eval_origins.size))
    cols = ["etiqueta", "mase", "mape", "mad", "me", "n_preds"]
    print(result.evaluation.ranked[cols].round(3).to_string(index=False))

    print("\n=== Ganador y ajuste de hiperparametros ===")
    winner = result.winner
    tun = result.tuning.get(winner)
    print("Metodo ganador: {}".format(winner))
    if tun is not None:
        print("Ajustado: {} | {}".format(tun.tuned, tun.reason))
        print("Parametros: {}".format(result.winner_params))

    print("\n=== Pronostico a 12 meses con intervalo del 95% ===")
    spec = get_spec(winner)
    pi = prediction_interval(s, spec, result.winner_params, season_length=12,
                             horizon=12, level=0.95)
    tabla = pi.to_frame().round(1)
    print(tabla.to_string())
    print("Metodo del intervalo: {}".format(pi.method))

    print("\n=== Politica de inventario (lead time = 3 meses, nivel de servicio 95%) ===")
    pol = compute_policy(s, spec, result.winner_params, lead_time=3,
                         service_level=0.95, season_length=12)
    print(pol.describe())
    for w in pol.warnings:
        print("  AVISO:", w)

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(out / "caso_ilustrativo_pronostico.csv", encoding="utf-8-sig")
    result.evaluation.ranked[cols].round(3).to_csv(
        out / "caso_ilustrativo_ranking.csv", index=False, encoding="utf-8-sig")
    print("\nCSV en:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
