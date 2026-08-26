"""Figuras de la comparativa externa (Fase 11, F28: "necesitamos mas graficos").

Archivo SEPARADO de `make_figures.py` a proposito: `make_figures.py` produce
las figuras del manuscrito original (Fases 1-9) y no se toca en esta fase
(restriccion de aislamiento, ver `docs/prompt_maestro.md` de la Fase 11).
Este script solo LEE `resultados/comparativa_externa.csv` (generado por
`comparativa_externa.py`) y regenera un puñado de series representativas
para las figuras que necesitan la trayectoria completa del pronostico, no
solo sus metricas resumidas -las metricas del CSV no alcanzan para dibujar
"historico + pronostico" superpuesto.

Misma carpeta de salida que `make_figures.py`
(`manuscritos/articulo_mdpi/figures/`) y el mismo estilo matplotlib.

Genera:
  fig_c1_boxplot_mase.png       - distribucion de MASE por metodo
  fig_c2_mase_vs_longitud.png   - precision (MASE) vs. longitud de serie
  fig_c3_panel_regimenes.png    - pequeños multiplos: historico + pronostico
                                   de los tres metodos, por regimen

Uso:
    python codigo/experimentos/comparativa_externa.py   # genera el CSV primero
    python codigo/experimentos/make_figures_comparativa.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from external_baselines.adapters import external_specs  # noqa: E402
from forecasting_core.models import get_spec  # noqa: E402
from forecasting_core.optimize import run_pipeline  # noqa: E402
from forecasting_core.validation import backtest_one_step, rolling_origins  # noqa: E402

from comparativa_externa import make_regime_series  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultados"
OUT = Path(__file__).resolve().parents[2] / "manuscritos" / "articulo_mdpi" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200,
})

# Paleta consistente entre las tres figuras y con make_figures.py.
COLOR_HISTORICO = "#2E3A8C"
COLOR_HERRAMIENTA = "#177245"
COLOR_PROPHET = "#B34700"
COLOR_LIGHTGBM = "#6A3D9A"
COLOR_NAIVE = "#8C8C8C"

METHOD_STYLE = {
    "herramienta": ("Herramienta (ganador honesto)", COLOR_HERRAMIENTA),
    "naive": ("Naive", COLOR_NAIVE),
    "seasonal_naive": ("Naive estacional", "#B0A000"),
    "prophet": ("Prophet", COLOR_PROPHET),
    "lightgbm": ("LightGBM (mlforecast)", COLOR_LIGHTGBM),
}


def _load_csv() -> pd.DataFrame:
    path = RESULTS_DIR / "comparativa_externa.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No existe {}. Corra primero: python codigo/experimentos/"
            "comparativa_externa.py".format(path)
        )
    df = pd.read_csv(path)
    return df[df["estado"] == "ok"].copy()


def _long_mase(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    cols = ["{}_mase".format(m) for m in methods]
    present = [c for c in cols if c in df.columns]
    long = df[["n", "regimen"] + present].melt(
        id_vars=["n", "regimen"], value_vars=present, var_name="metodo_col", value_name="mase"
    )
    long["metodo"] = long["metodo_col"].str.replace("_mase", "", regex=False)
    return long.dropna(subset=["mase"])


def fig_c1_boxplot_mase(df: pd.DataFrame):
    """Distribucion de MASE por metodo (F28): Herramienta/Prophet/LightGBM/naive."""
    methods = [m for m in ("herramienta", "naive", "seasonal_naive", "prophet", "lightgbm")]
    long = _long_mase(df, methods)
    present_methods = [m for m in methods if m in long["metodo"].unique()]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    data = [long.loc[long["metodo"] == m, "mase"].to_numpy() for m in present_methods]
    labels = [METHOD_STYLE[m][0] for m in present_methods]
    colors = [METHOD_STYLE[m][1] for m in present_methods]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                                   markersize=5),
                    medianprops=dict(color="black", linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.6)
    ax.annotate("MASE=1 (empata con el naive estacional in-sample)", xy=(0.985, 1.0),
               xycoords=("axes fraction", "data"), ha="right", va="bottom",
               fontsize=7.5, color="red")
    ax.set_ylabel("MASE (bloque externo, protocolo honesto)")
    ax.set_title("Distribucion de MASE por metodo — {} series sinteticas\n"
                 "({} longitudes x {} regimenes estructurales)".format(
                     len(df), df["n"].nunique(), df["regimen"].nunique()),
               fontsize=9.6)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c1_boxplot_mase.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_c1: fig_c1_boxplot_mase.png")


def fig_c2_mase_vs_longitud(df: pd.DataFrame):
    """Precision (MASE) vs. longitud de serie, un color por metodo (F28)."""
    methods = ["herramienta", "prophet", "lightgbm"]
    long = _long_mase(df, methods)
    present_methods = [m for m in methods if m in long["metodo"].unique()]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for m in present_methods:
        sub = long[long["metodo"] == m]
        label, color = METHOD_STYLE[m]
        ax.scatter(sub["n"], sub["mase"], color=color, alpha=0.35, s=26, zorder=2)
        agg = sub.groupby("n")["mase"].median().reset_index()
        ax.plot(agg["n"], agg["mase"], "-o", color=color, lw=1.8, ms=5,
               label="{} (mediana por n)".format(label), zorder=3)

    ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Longitud de la serie (observaciones mensuales)")
    ax.set_ylabel("MASE (bloque externo)")
    ax.set_title("Precision vs. longitud de serie por metodo\n"
                 "(puntos: series individuales; linea: mediana por longitud)", fontsize=9.6)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c2_mase_vs_longitud.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_c2: fig_c2_mase_vs_longitud.png")


def _fit_full_history_forecasts(s: pd.Series, h: int, specs: dict) -> dict:
    """Pronostico de `h` pasos con historia COMPLETA, para dibujar (no metrica)."""
    out = {}
    try:
        res = run_pipeline(s, m=12)
        if res.ok and res.winner is not None:
            spec = get_spec(res.winner)
            fc = spec.forecast(s.to_numpy(dtype=float), params=res.winner_params, h=h, m=12)
            out["herramienta"] = (res.winner, fc)
    except Exception as exc:
        print("  [aviso] herramienta fallo en el panel de pequeños multiplos: {}".format(exc))

    y = s.to_numpy(dtype=float)
    for key, spec in specs.items():
        short = key.replace("ext_", "")
        try:
            fc = spec.forecast(y, params=None, h=h, m=12)
            out[short] = (short, fc)
        except Exception as exc:
            print("  [aviso] {} fallo en el panel de pequeños multiplos: {}".format(short, exc))
    return out


def fig_c3_panel_regimenes(seed: int = 20260824, h: int = 12):
    """Pequeños multiplos: historico + pronostico de los TRES metodos a la vez.

    Formato inspirado en el PDF de los tutores (dual historico+pronostico),
    pero con protocolo correcto: los tres metodos en el MISMO panel, no de a
    dos, y usando la historia completa (esto es una figura ILUSTRATIVA del
    pronostico final que produciria cada metodo, no la evaluacion de
    desempeño -esa es `fig_c1`/`fig_c2`, calculada honestamente sobre el
    bloque externo).
    """
    specs = external_specs()
    if not specs:
        print("  [omitido] fig_c3: ni prophet ni lightgbm estan instalados en este entorno.")
        return

    regimenes = [
        (24, "corta_erratica", "Corto y erratico (n=24, el caso que rompio a\n"
                                "Prophet mal configurado en el PDF de los tutores)"),
        (48, "tendencia", "Tendencia (n=48)"),
        (48, "estacional", "Estacional (n=48)"),
        (60, "tendencia_estacional", "Tendencia + estacionalidad (n=60)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    for ax, (n, regimen, titulo) in zip(axes.ravel(), regimenes):
        s = make_regime_series(n, regimen, seed)
        forecasts = _fit_full_history_forecasts(s, h, specs)

        ax.plot(s.index, s.values, "-o", color=COLOR_HISTORICO, ms=3, lw=1.2,
               label="Historico", zorder=2)
        future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        ax.axvline(s.index[-1], color="gray", ls=":", lw=1)

        for key in ("herramienta", "prophet", "lightgbm"):
            if key not in forecasts:
                continue
            detalle, fc = forecasts[key]
            label, color = METHOD_STYLE[key]
            if key == "herramienta":
                label = "{} ({})".format(label, detalle)
            ax.plot(future_idx, fc, "-o", color=color, ms=3.4, lw=1.6, label=label, zorder=3)

        ax.set_title(titulo, fontsize=9.4)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7.6)
        ax.legend(fontsize=6.6, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "Historico + pronostico a {} meses: Herramienta vs. Prophet vs. LightGBM\n"
        "(pronostico ilustrativo con historia completa; el desempeño honesto por "
        "regimen esta en fig_c1/fig_c2)".format(h),
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_c3_panel_regimenes.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_c3: fig_c3_panel_regimenes.png")


def main():
    df = _load_csv()
    fig_c1_boxplot_mase(df)
    fig_c2_mase_vs_longitud(df)
    fig_c3_panel_regimenes()
    print("\nFiguras en:", OUT)


if __name__ == "__main__":
    main()
