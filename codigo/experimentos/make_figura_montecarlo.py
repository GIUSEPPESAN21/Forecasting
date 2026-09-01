"""Figura 2 del manuscrito: validacion Monte Carlo de las pruebas de
clasificacion estructural (tendencia y estacionalidad).

Archivo SEPARADO de `make_figures.py` y de `make_figures_comparativa.py` a
proposito, siguiendo la misma restriccion de aislamiento por fase que ya usan
esos dos scripts: este solo LEE `resultados/montecarlo_clasificacion.csv`
(generado por `montecarlo_clasificacion.py`, que no se modifica aqui) y
dibuja los tres paneles que el manuscrito describe en la Seccion 3.2:

  Panel A: tasa de falso positivo de la prueba de TENDENCIA adoptada, por
           escenario (con tendencia verdadera = False) y longitud de serie.
  Panel B: tasa de falso positivo de la prueba de ESTACIONALIDAD adoptada,
           por escenario (con estacionalidad verdadera = False) y longitud
           (n=24 se omite: la estacionalidad no es evaluable con menos de
           tres ciclos anuales completos, Seccion 2.4).
  Panel C: tasa agregada (media sobre escenarios y longitudes) del atajo
           comun de cada prueba (p-valor de una regresion sin ponderar para
           tendencia; umbral de autocorrelacion cruda en el rezago 12 para
           estacionalidad) frente a la prueba adoptada en este pipeline. Los
           dos valores del atajo comun (74.4%, 50.2%) y de la prueba
           adoptada (8.9%, 1.4%) son exactamente los que ya imprime
           `montecarlo_clasificacion.py` en su resumen final (ver
           `resultados/logs/montecarlo_clasificacion.log`); este script los
           reproduce en lugar de recalcularlos para no duplicar logica.

Uso:
    python codigo/experimentos/montecarlo_clasificacion.py   # genera el CSV primero
    python codigo/experimentos/make_figura_montecarlo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultados"
OUT = Path(__file__).resolve().parents[2] / "manuscritos" / "articulo_mdpi" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans",
    "figure.dpi": 200, "savefig.dpi": 200,
})

# Etiquetas de escenario para el eje del panel A y B (orden fijo, coincide
# con la Figura 2 del manuscrito).
TREND_FALSE_SCENARIOS = [
    ("AR(1) phi=0.7", "AR(1) $\\phi$=0.7"),
    ("estacional pura", "Pure seasonal"),
    ("paseo aleatorio", "Random walk"),
    ("ruido blanco", "White noise"),
]
SEASONAL_FALSE_SCENARIOS = [
    ("AR(1) phi=0.7", "AR(1) $\\phi$=0.7"),
    ("tendencia limpia", "Clean trend"),
    ("paseo con deriva", "Drift walk"),
    ("tendencia lineal", "Linear trend"),
    ("paseo aleatorio", "Random walk"),
    ("ruido blanco", "White noise"),
]

# Reproducidos textualmente del resumen impreso por montecarlo_clasificacion.py
# (resultados/logs/montecarlo_clasificacion.log): tasa de falso positivo del
# atajo comun de cada prueba, frente al atajo adoptado en este pipeline.
COMMON_SHORTCUT_TREND_PCT = 74.4
ADOPTED_TREND_MEAN_PCT = 8.9
COMMON_SHORTCUT_SEASONAL_PCT = 50.2
ADOPTED_SEASONAL_MEAN_PCT = 1.4


def _load() -> pd.DataFrame:
    path = RESULTS_DIR / "montecarlo_clasificacion.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No existe {}. Corra primero: python codigo/experimentos/"
            "montecarlo_clasificacion.py".format(path)
        )
    return pd.read_csv(path)


def _heatmap_matrix(df: pd.DataFrame, scenarios, value_col: str, lengths):
    mat = np.full((len(scenarios), len(lengths)), np.nan)
    for i, (serie, _) in enumerate(scenarios):
        sub = df[df["serie"] == serie]
        for j, n in enumerate(lengths):
            row = sub[sub["n"] == n]
            if len(row) and pd.notna(row[value_col].iloc[0]):
                mat[i, j] = 100.0 * float(row[value_col].iloc[0])
    return mat


def make_figure() -> Path:
    df = _load()

    lengths_trend = [24, 36, 48, 120]
    lengths_seasonal = [36, 48, 120]

    mat_a = _heatmap_matrix(df, TREND_FALSE_SCENARIOS, "error_tendencia", lengths_trend)
    mat_b = _heatmap_matrix(df, SEASONAL_FALSE_SCENARIOS, "error_estacionalidad", lengths_seasonal)

    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 0.85], wspace=0.55)

    # --- Panel A: trend test ---
    axA = fig.add_subplot(gs[0, 0])
    imA = axA.imshow(mat_a, cmap="YlOrRd", vmin=0, vmax=20, aspect="auto")
    axA.set_xticks(range(len(lengths_trend)))
    axA.set_xticklabels(lengths_trend)
    axA.set_yticks(range(len(TREND_FALSE_SCENARIOS)))
    axA.set_yticklabels([lbl for _, lbl in TREND_FALSE_SCENARIOS])
    axA.set_xlabel("n (observations)")
    axA.set_title("(A) Trend test: false-positive rate (%)\nby scenario and series length (adopted test)",
                   fontsize=8)
    for i in range(mat_a.shape[0]):
        for j in range(mat_a.shape[1]):
            if not np.isnan(mat_a[i, j]):
                axA.text(j, i, "{:.1f}".format(mat_a[i, j]), ha="center", va="center", fontsize=7.5)
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)

    # --- Panel B: seasonality test ---
    axB = fig.add_subplot(gs[0, 1])
    imB = axB.imshow(mat_b, cmap="YlOrRd", vmin=0, vmax=20, aspect="auto")
    axB.set_xticks(range(len(lengths_seasonal)))
    axB.set_xticklabels(lengths_seasonal)
    axB.set_yticks(range(len(SEASONAL_FALSE_SCENARIOS)))
    axB.set_yticklabels([lbl for _, lbl in SEASONAL_FALSE_SCENARIOS])
    axB.set_xlabel("n (observations)")
    axB.set_title("(B) Seasonality test: false-positive rate (%)\nby scenario and series length (adopted test)",
                   fontsize=8)
    for i in range(mat_b.shape[0]):
        for j in range(mat_b.shape[1]):
            if not np.isnan(mat_b[i, j]):
                axB.text(j, i, "{:.1f}".format(mat_b[i, j]), ha="center", va="center", fontsize=7.5)
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)

    # --- Panel C: aggregate common shortcut vs adopted test ---
    axC = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    width = 0.32
    axC.bar(x - width / 2, [COMMON_SHORTCUT_TREND_PCT, COMMON_SHORTCUT_SEASONAL_PCT], width,
            label="Common shortcut", color="#B22222")
    axC.bar(x + width / 2, [ADOPTED_TREND_MEAN_PCT, ADOPTED_SEASONAL_MEAN_PCT], width,
            label="Adopted test", color="#1B7837")
    axC.axhline(5.0, color="black", linestyle=":", linewidth=1.0)
    axC.set_xticks(x)
    axC.set_xticklabels(["Trend test", "Seasonality test"])
    axC.set_ylabel("Mean false-positive rate (%)")
    axC.set_title("(C) Aggregate false-positive rate:\ncommon shortcut vs. adopted test", fontsize=8)
    for xi, v in zip(x - width / 2, [COMMON_SHORTCUT_TREND_PCT, COMMON_SHORTCUT_SEASONAL_PCT]):
        axC.text(xi, v + 1.0, "{:.1f}".format(v), ha="center", fontsize=7.5)
    for xi, v in zip(x + width / 2, [ADOPTED_TREND_MEAN_PCT, ADOPTED_SEASONAL_MEAN_PCT]):
        axC.text(xi, v + 1.0, "{:.1f}".format(v), ha="center", fontsize=7.5)
    axC.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Monte Carlo validation of the structural classification tests (1000 replicates/scenario, seed 20260824)",
        fontsize=9.5, y=1.03,
    )

    out_path = OUT / "fig2_montecarlo_clasificacion.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    path = make_figure()
    print("Figura:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
