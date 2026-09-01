"""Sensibilidad de la comparacion externa a la longitud de serie, e
intervalos de confianza bootstrap sobre sus estadisticos principales.

Motivacion
----------
La Tabla 11/12 del manuscrito (ver `comparativa_externa.py`) agrega 50 series
sinteticas (10 longitudes x 5 regimenes) en una sola mediana de MASE por
metodo. La Figura 5 (`fig_c2_mase_vs_longitud.png`) muestra que la mediana de
LightGBM cae a su valor mas bajo del barrido en n=120, lo que podria leerse
como evidencia de que el comparador global se vuelve mejor con series mas
largas. Con solo 5 series por longitud, esa lectura no es confiable sin una
prueba: este script responde dos preguntas que la Tabla 11 no puede
responder por si sola:

1. ¿La longitud de la serie esta relacionada, de forma monotona, con el MASE
   de cada metodo o con la ventaja de la Herramienta sobre cada comparador?
   Se usa una correlacion de rangos de Spearman entre `n` y (a) el MASE de
   cada metodo, (b) la diferencia pareada Herramienta-menos-comparador.
2. ¿Que tan preciso es cada estadistico agregado de la Tabla 11 (mediana de
   MASE por metodo, tasa de victoria de la Herramienta contra cada
   comparador)? Se reporta un intervalo de confianza bootstrap percentil
   (10000 remuestreos, semilla fija) para cada uno.

Este script REUTILIZA `resultados/comparativa_externa.csv` (no vuelve a
correr el pipeline ni genera series nuevas): es un analisis puramente
estadistico sobre un artefacto que ya existe, consistente con la Tabla 5 del
manuscrito ("Reuses comparativa_externa.csv").

Uso
---
    python codigo/experimentos/analisis_sensibilidad_ci.py
    python codigo/experimentos/analisis_sensibilidad_ci.py --seed 20260824 --resamples 10000

Salida
------
`resultados/analisis_sensibilidad_ci.csv` (una fila por estadistico, con su
IC del 95%) y `manuscritos/articulo_mdpi/figures/fig_c4_bootstrap_winrate.png`
(Figura 7 del manuscrito: distribucion bootstrap de la tasa de victoria de la
Herramienta contra Prophet y contra LightGBM).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTADOS_DIR = Path(__file__).resolve().parents[2] / "resultados"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "manuscritos" / "articulo_mdpi" / "figures"


def _load_comparativa() -> pd.DataFrame:
    path = RESULTADOS_DIR / "comparativa_externa.csv"
    if not path.exists():
        raise SystemExit(
            "No se encontro {}. Corra primero codigo/experimentos/comparativa_externa.py".format(path)
        )
    df = pd.read_csv(path)
    ok = df[
        (df["estado"] == "ok")
        & (df.get("prophet_estado") == "ok")
        & (df.get("lightgbm_estado") == "ok")
    ].dropna(subset=["herramienta_mase", "prophet_mase", "lightgbm_mase"]).copy()
    return ok


def spearman_block(df: pd.DataFrame) -> list[dict]:
    """Correlacion de Spearman entre n y (MASE por metodo, diferencias pareadas)."""
    rows = []
    for label, col in (
        ("herramienta_mase", "herramienta_mase"),
        ("prophet_mase", "prophet_mase"),
        ("lightgbm_mase", "lightgbm_mase"),
    ):
        rho, p = stats.spearmanr(df["n"], df[col])
        rows.append({"estadistico": "spearman_n_vs_{}".format(label), "rho": rho, "p_valor": p, "n": len(df)})

    diff_prophet = df["herramienta_mase"] - df["prophet_mase"]
    diff_lightgbm = df["herramienta_mase"] - df["lightgbm_mase"]
    for label, diff in (
        ("herramienta_menos_prophet", diff_prophet),
        ("herramienta_menos_lightgbm", diff_lightgbm),
    ):
        rho, p = stats.spearmanr(df["n"], diff)
        rows.append({"estadistico": "spearman_n_vs_{}".format(label), "rho": rho, "p_valor": p, "n": len(df)})
    return rows


def _bootstrap_percentile_ci(values: np.ndarray, statistic, resamples: int, rng: np.random.Generator,
                              alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap percentil simple: remuestrea `values` con reemplazo `resamples` veces."""
    n = len(values)
    boot = np.empty(resamples)
    idx_all = rng.integers(0, n, size=(resamples, n))
    for i in range(resamples):
        boot[i] = statistic(values[idx_all[i]])
    point = statistic(values)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, boot


def bootstrap_block(df: pd.DataFrame, resamples: int, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    rows = []
    boot_dists = {}

    for label, col in (("herramienta", "herramienta_mase"), ("prophet", "prophet_mase"),
                        ("lightgbm", "lightgbm_mase")):
        point, lo, hi, _ = _bootstrap_percentile_ci(df[col].to_numpy(), np.median, resamples, rng)
        rows.append({"estadistico": "mediana_mase_{}".format(label), "punto": point,
                     "ci95_lo": lo, "ci95_hi": hi, "n": len(df)})

    for ext in ("prophet", "lightgbm"):
        h = df["herramienta_mase"].to_numpy()
        c = df["{}_mase".format(ext)].to_numpy()
        diff = h - c
        win = (h < c).astype(float)

        point, lo, hi, _ = _bootstrap_percentile_ci(diff, np.median, resamples, rng)
        rows.append({"estadistico": "diferencia_mediana_herramienta_menos_{}".format(ext), "punto": point,
                     "ci95_lo": lo, "ci95_hi": hi, "n": len(df)})

        point, lo, hi, dist = _bootstrap_percentile_ci(win, np.mean, resamples, rng)
        rows.append({"estadistico": "tasa_victoria_herramienta_vs_{}".format(ext), "punto": point,
                     "ci95_lo": lo, "ci95_hi": hi, "n": len(df)})
        boot_dists[ext] = dist

    return rows, boot_dists


def make_figure(boot_dists: dict, observed: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "figure.dpi": 200, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"prophet": "#d97a2b", "lightgbm": "#6a4c93"}
    labels = {
        "prophet": "vs. Prophet (observed {:.0%}, 95% CI [{:.0%}, {:.0%}])".format(
            observed["prophet"][0], observed["prophet"][1], observed["prophet"][2]),
        "lightgbm": "vs. LightGBM (observed {:.0%}, 95% CI [{:.0%}, {:.0%}])".format(
            observed["lightgbm"][0], observed["lightgbm"][1], observed["lightgbm"][2]),
    }
    bins = np.linspace(0.0, 1.0, 41)
    for ext, dist in boot_dists.items():
        ax.hist(dist, bins=bins, alpha=0.55, density=True, color=colors[ext], label=labels[ext])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="Chance level (50%)")
    ax.set_xlabel("Bootstrap win rate (tool MASE < comparator MASE)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap distribution of the tool's win rate\nagainst Prophet and LightGBM (10,000 resamples)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()

    df = _load_comparativa()
    print("Series con Herramienta+Prophet+LightGBM validas: {}".format(len(df)))
    if len(df) == 0:
        raise SystemExit("Sin filas validas en comparativa_externa.csv; corra ese script primero.")

    rows = []
    rows += spearman_block(df)
    boot_rows, boot_dists = bootstrap_block(df, args.resamples, args.seed)
    rows += boot_rows

    out = pd.DataFrame(rows)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTADOS_DIR / "analisis_sensibilidad_ci.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("CORRELACION DE SPEARMAN (n vs. MASE / diferencias pareadas)")
    print("=" * 88)
    for r in rows:
        if r["estadistico"].startswith("spearman"):
            print("  {:42s} rho={:+.3f}  p={:.3f}  n={}".format(r["estadistico"], r["rho"], r["p_valor"], r["n"]))

    print("\n" + "=" * 88)
    print("INTERVALOS DE CONFIANZA BOOTSTRAP (percentil 95%, {} remuestreos, seed={})".format(
        args.resamples, args.seed))
    print("=" * 88)
    observed = {}
    for r in rows:
        if "estadistico" in r and (r["estadistico"].startswith("mediana_mase")
                                    or r["estadistico"].startswith("diferencia_mediana")
                                    or r["estadistico"].startswith("tasa_victoria")):
            print("  {:42s} punto={:.3f}  95% CI [{:.3f}, {:.3f}]".format(
                r["estadistico"], r["punto"], r["ci95_lo"], r["ci95_hi"]))
        if r["estadistico"].startswith("tasa_victoria_herramienta_vs_"):
            ext = r["estadistico"].replace("tasa_victoria_herramienta_vs_", "")
            observed[ext] = (r["punto"], r["ci95_lo"], r["ci95_hi"])

    fig_path = FIGURES_DIR / "fig_c4_bootstrap_winrate.png"
    make_figure(boot_dists, observed, fig_path)

    print("\nCSV:", out_path)
    print("Figura:", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
