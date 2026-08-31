"""Compara el panel publico M3 evaluado a distintas longitudes (F36).

`panel_publico.py` evalua un panel de M3 mensual truncado a `--max-len`
observaciones por serie; la corrida original solo se hizo con `--max-len 48`.
Este script pequeño no vuelve a correr el pipeline: lee los CSV que ya
produjo `panel_publico.py --max-len {24,36,48}` (mismo panel base, misma
semilla) y arma una tabla resumen con la mediana de MASE de la herramienta
por longitud, para poner en un solo lugar lo que hoy exige abrir tres CSV.

Uso (tras correr panel_publico.py con cada longitud):
    python experiments/panel_publico.py --max-len 24 --n-series 150 --seed 20260824
    python experiments/panel_publico.py --max-len 36 --n-series 150 --seed 20260824
    python experiments/panel_publico.py --max-len 48 --n-series 150 --seed 20260824
    python experiments/comparar_longitudes_panel.py --lengths 24 36 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 36, 48])
    args = ap.parse_args()

    rows = []
    for length in args.lengths:
        path = OUT_DIR / "panel_publico_len{}.csv".format(length)
        if not path.exists():
            print("  [omitido] {} no existe -- corra panel_publico.py --max-len {} primero"
                  .format(path.name, length))
            continue
        df = pd.read_csv(path)
        comp = df.dropna(subset=["mase_ganador", "mase_naive"])
        tasa = (comp["mase_ganador"] < comp["mase_naive"]).mean() if len(comp) else float("nan")
        rows.append({
            "max_len": length, "n_series": len(df), "n_pares_validos": len(comp),
            "mase_ganador_mediana": float(df["mase_ganador"].median()),
            "mase_naive_mediana": float(df["mase_naive"].median()) if "mase_naive" in df else float("nan"),
            "tasa_victoria_vs_naive": float(tasa),
        })

    if not rows:
        print("Ningun CSV de longitud disponible; nada que comparar.")
        return 1

    resumen = pd.DataFrame(rows)
    resumen.to_csv(OUT_DIR / "panel_publico_comparacion_longitudes.csv", index=False,
                    encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("COMPARACION DEL PANEL PUBLICO M3 POR LONGITUD (max_len)")
    print("=" * 88)
    for _, r in resumen.iterrows():
        print("  max_len={:3.0f}  n={:4.0f}  MASE herramienta mediano={:.3f}  "
              "MASE naive mediano={:.3f}  victoria vs naive={:.1%}".format(
                  r["max_len"], r["n_series"], r["mase_ganador_mediana"],
                  r["mase_naive_mediana"], r["tasa_victoria_vs_naive"]))
    print("=" * 88)
    print("\nCSV:", OUT_DIR / "panel_publico_comparacion_longitudes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
