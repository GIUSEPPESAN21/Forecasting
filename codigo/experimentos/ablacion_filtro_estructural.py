"""Ablacion del filtro estructural (F33).

La Introduccion del manuscrito pregunta cuanto del desempeno de la
herramienta se debe al filtrado estructural (`structural_filter` en
`honest_outer_estimate`, que descarta de antemano modelos estacionales sobre
series sin estacionalidad detectada, y viceversa) frente a una comparacion
sin esa restriccion, donde `run_pipeline` elige entre TODOS los modelos por
validacion cruzada sin ningun filtro previo. No existia el experimento que
respondiera esa pregunta con una cifra.

Este script corre el MISMO panel de M3 mensual truncado (misma semilla, mismo
`--max-len`) que usa `panel_publico.py`, dos veces sobre las MISMAS series:
una con `structural_filter=True` (comportamiento por defecto de la
herramienta) y otra con `structural_filter=False` (el pipeline elige
libremente entre todos los modelos candidatos). No se toca la logica interna
de `honest_outer_estimate`/`eligible_specs`: solo se le pasa el parametro que
ya expone.

Uso
---
    python experiments/ablacion_filtro_estructural.py --n-series 150 --max-len 48 --seed 20260824
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from panel_publico import evaluate_one, load_m3_monthly_short  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"


def run_config(ids, df: pd.DataFrame, structural_filter: bool) -> tuple[pd.DataFrame, float]:
    rows = []
    t0 = time.perf_counter()
    for i, uid in enumerate(ids, 1):
        y = df.loc[df["unique_id"] == uid, "y"].to_numpy(dtype=float)
        row = evaluate_one(uid, y, structural_filter=structural_filter)
        rows.append(row)
        if i % 25 == 0 or i == len(ids):
            print("  [filtro={}] [{}/{}] procesadas".format(structural_filter, i, len(ids)))
    elapsed = time.perf_counter() - t0
    return pd.DataFrame(rows), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-series", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    print("Descargando/leyendo M3 mensual (datasetsforecast)...")
    df = load_m3_monthly_short(args.max_len, args.min_len)
    todas_ids = df["unique_id"].unique()
    if len(todas_ids) == 0:
        print("No hay series elegibles; revise --min-len / --max-len.")
        return 1

    rng = np.random.default_rng(args.seed)
    ids = rng.choice(todas_ids, size=min(args.n_series, len(todas_ids)), replace=False)
    print("Muestra evaluada: {} series (semilla {}) -- MISMAS series en ambos modos"
          .format(len(ids), args.seed))

    res_on, t_on = run_config(ids, df, structural_filter=True)
    res_off, t_off = run_config(ids, df, structural_filter=False)

    ok_on = res_on[res_on["estado"] == "ok"].copy()
    ok_off = res_off[res_off["estado"] == "ok"].copy()
    ok_on["structural_filter"] = True
    ok_off["structural_filter"] = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([ok_on, ok_off], ignore_index=True)
    combined.to_csv(OUT_DIR / "ablacion_filtro_estructural.csv", index=False, encoding="utf-8-sig")

    def resumen(ok: pd.DataFrame) -> dict:
        comp = ok.dropna(subset=["mase_ganador", "mase_naive"])
        supera = (comp["mase_ganador"] < comp["mase_naive"]).mean() if len(comp) else float("nan")
        return {
            "n_ok": len(ok),
            "n_pares_validos": len(comp),
            "mase_ganador_mediana": float(ok["mase_ganador"].median()) if "mase_ganador" in ok else float("nan"),
            "tasa_victoria_vs_naive": float(supera),
        }

    r_on, r_off = resumen(ok_on), resumen(ok_off)
    diff_mediana = r_on["mase_ganador_mediana"] - r_off["mase_ganador_mediana"]
    diff_tasa = r_on["tasa_victoria_vs_naive"] - r_off["tasa_victoria_vs_naive"]

    print("\n" + "=" * 88)
    print("ABLACION DEL FILTRO ESTRUCTURAL ({} series identicas en ambos modos)".format(len(ids)))
    print("=" * 88)
    print("  structural_filter=True  : n_ok={n_ok:3d}  MASE mediano={mase_ganador_mediana:.3f}  "
          "victoria vs naive={tasa_victoria_vs_naive:.1%}  tiempo={t:.1f}s".format(**r_on, t=t_on))
    print("  structural_filter=False : n_ok={n_ok:3d}  MASE mediano={mase_ganador_mediana:.3f}  "
          "victoria vs naive={tasa_victoria_vs_naive:.1%}  tiempo={t:.1f}s".format(**r_off, t=t_off))
    print("  Diferencia (on - off)   : MASE mediano {:+.3f}  tasa de victoria {:+.1%}  "
          "tiempo {:+.1f}s".format(diff_mediana, diff_tasa, t_on - t_off))

    resumen_df = pd.DataFrame([
        {"structural_filter": True, "tiempo_s": t_on, **r_on},
        {"structural_filter": False, "tiempo_s": t_off, **r_off},
    ])
    resumen_df.to_csv(OUT_DIR / "ablacion_filtro_estructural_resumen.csv", index=False,
                       encoding="utf-8-sig")
    print("\nCSV detalle:", OUT_DIR / "ablacion_filtro_estructural.csv")
    print("CSV resumen:", OUT_DIR / "ablacion_filtro_estructural_resumen.csv")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
