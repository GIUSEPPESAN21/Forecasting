"""Genera las figuras reales del manuscrito (Fase 9 / F09 en template.tex).

Nunca reutiliza la misma imagen (flowchart_tool.png) diez veces como el
manuscrito original: cada figura se produce a partir de los datos reales de
`experiments/` o describe la arquitectura corregida.

Uso:
    python experiments/make_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})


def fig1_flowchart():
    """Fig. 1: flujo metodologico corregido (reemplaza el placeholder reusado 10 veces)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    stages = [
        (0.4, 5.2, "1. Data upload &\nvalidation", "#2E3A8C"),
        (0.4, 3.8, "2. Sequential structural\nclassification\n(seasonality → trend →\nstationarity)", "#2E3A8C"),
        (0.4, 2.2, "3. Candidate pool\n(structural filter)", "#3D4FA8"),
        (0.4, 0.7, "4. Walk-forward:\ntuning block", "#5A6BC0"),
    ]
    stage2 = [
        (5.4, 0.7, "4. Walk-forward:\nEVALUATION block\n(held out from tuning)", "#B34700"),
        (5.4, 2.2, "5. Ranking by MASE\n(naive/snaive always\nincluded)", "#B34700"),
        (5.4, 3.8, "6. Forecast + empirical\nprediction interval", "#177245"),
        (5.4, 5.2, "7. Inventory policy\n(safety stock, ROP)", "#177245"),
    ]

    def box(x, y, text, color):
        b = FancyBboxPatch((x, y - 0.55), 3.9, 1.1, boxstyle="round,pad=0.08,rounding_size=0.12",
                           linewidth=1.4, edgecolor=color, facecolor="white", zorder=2)
        ax.add_patch(b)
        ax.text(x + 1.95, y, text, ha="center", va="center", fontsize=8.6, color="#1a1a1a", zorder=3)

    for x, y, t, c in stages + stage2:
        box(x, y, t, c)

    def arrow(p0, p1, color="#555555"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.3, color=color, zorder=1))

    arrow((2.35, 4.65), (2.35, 4.35))
    arrow((2.35, 3.25), (2.35, 2.75))
    arrow((2.35, 1.65), (2.35, 1.25))
    arrow((4.3, 0.7), (5.4, 0.7))
    arrow((7.35, 1.25), (7.35, 1.65))
    arrow((7.35, 2.75), (7.35, 3.25))
    arrow((7.35, 4.35), (7.35, 4.65))

    ax.annotate("Never re-used to score\nhyperparameters", xy=(5.4, 0.7), xytext=(3.6, -0.35),
               fontsize=7.3, color="#B34700", ha="center",
               arrowprops=dict(arrowstyle="->", color="#B34700", lw=0.8))
    ax.text(5.0, 6.05, "Corrected pipeline: seasonality tested before trend, hyperparameter\n"
                       "tuning strictly separated from the reported evaluation block.",
           ha="center", fontsize=8.8, style="italic", color="#333333")

    fig.tight_layout()
    fig.savefig(OUT / "flowchart_tool.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1: flowchart_tool.png")


def fig2_forecast_caso():
    """Fig. 2: pronostico con intervalo para el caso ilustrativo (Sec. 3.2)."""
    from forecasting_core.intervals import prediction_interval
    from forecasting_core.models import get_spec
    from forecasting_core.optimize import run_pipeline

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from caso_ilustrativo import make_case_series  # noqa: E402

    s = make_case_series()
    res = run_pipeline(s, m=12)
    spec = get_spec(res.winner)
    pi = prediction_interval(s, spec, res.winner_params, season_length=12, horizon=12, level=0.95)

    fit_dates = res.evaluation.dates
    fit_pred = res.evaluation.backtests[res.winner].y_pred

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(s.index, s.values, "-o", color="#2E3A8C", ms=3.5, lw=1.4, label="Historical demand")
    ax.plot(fit_dates, fit_pred, "s", color="#B34700", ms=5, mfc="none", mew=1.4,
           label="Out-of-sample fitted (evaluation block)")
    ax.plot(pi.index, pi.mean, "-o", color="#177245", ms=3.5, lw=1.6, label="12-month forecast")
    ax.fill_between(pi.index, pi.lower, pi.upper, color="#177245", alpha=0.15,
                    label="95% prediction interval")
    ax.axvline(s.index[-1], color="gray", ls=":", lw=1)
    ax.set_ylabel("Demand (units)")
    ax.set_title("Illustrative case: historical demand, out-of-sample fit, and forecast\n"
                 "(method: {})".format(res.winner), fontsize=10)
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_forecast_caso.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2: fig2_forecast_caso.png")


def main():
    fig1_flowchart()
    fig2_forecast_caso()
    print("\nFiguras en:", OUT)


if __name__ == "__main__":
    main()
