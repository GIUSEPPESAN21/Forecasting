"""Sensibilidad del bloque externo (`outer_block`) (F39).

La Seccion 3.5 del manuscrito advierte que la varianza de un MASE estimado
sobre 6-18 origenes es grande, pero cada MASE reportado en las Tablas 3-4 usa
un unico valor fijo: 6 origenes externos. Este script mide cuanto cambia el
MASE mediano y la tasa de victoria contra naive al variar
`outer_block in {6, 9, 12}` sobre el mismo panel de M3 mensual truncado que
usa `panel_publico.py`, para las series cuya longitud lo permite.

No se toca la logica interna de `honest_outer_estimate`: solo se le pasa un
`outer_block` distinto. Una serie que no tiene longitud suficiente para un
`outer_block` dado simplemente no produce ganador (`ok=False`) y se excluye
de ESE valor, sin cobertura silenciosa -el numero de series evaluadas por
cada `outer_block` se reporta explicitamente.

Uso
---
    python experiments/sensibilidad_outer_block.py --n-series 150 --max-len 48 --seed 20260824
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from panel_publico import evaluate_one, load_m3_monthly_short  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"
OUTER_BLOCKS = (6, 9, 12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-series", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--outer-blocks", type=int, nargs="+", default=list(OUTER_BLOCKS))
    args = ap.parse_args()

    print("Descargando/leyendo M3 mensual (datasetsforecast)...")
    df = load_m3_monthly_short(args.max_len, args.min_len)
    todas_ids = df["unique_id"].unique()
    if len(todas_ids) == 0:
        print("No hay series elegibles; revise --min-len / --max-len.")
        return 1

    rng = np.random.default_rng(args.seed)
    ids = rng.choice(todas_ids, size=min(args.n_series, len(todas_ids)), replace=False)
    print("Muestra base: {} series (semilla {}), evaluadas con outer_block in {}"
          .format(len(ids), args.seed, args.outer_blocks))

    all_rows = []
    resumen_rows = []
    for ob in args.outer_blocks:
        rows = []
        for i, uid in enumerate(ids, 1):
            y = df.loc[df["unique_id"] == uid, "y"].to_numpy(dtype=float)
            row = evaluate_one(uid, y, outer_block=ob)
            rows.append(row)
            if i % 25 == 0 or i == len(ids):
                print("  [outer_block={}] [{}/{}] procesadas".format(ob, i, len(ids)))
        res = pd.DataFrame(rows)
        ok = res[res["estado"] == "ok"].copy()
        ok["outer_block"] = ob
        all_rows.append(ok)

        comp = ok.dropna(subset=["mase_ganador", "mase_naive"])
        mediana = float(ok["mase_ganador"].median()) if len(ok) else float("nan")
        tasa = float((comp["mase_ganador"] < comp["mase_naive"]).mean()) if len(comp) else float("nan")
        resumen_rows.append({
            "outer_block": ob, "n_series_muestra": len(ids), "n_ok": len(ok),
            "n_excluidas_longitud_insuficiente": len(ids) - len(ok),
            "n_pares_validos": len(comp),
            "mase_ganador_mediana": mediana, "tasa_victoria_vs_naive": tasa,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(
        OUT_DIR / "sensibilidad_outer_block_detalle.csv", index=False, encoding="utf-8-sig")
    resumen_df = pd.DataFrame(resumen_rows)
    resumen_df.to_csv(OUT_DIR / "sensibilidad_outer_block.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("SENSIBILIDAD DEL BLOQUE EXTERNO (outer_block)")
    print("=" * 88)
    for r in resumen_rows:
        print("  outer_block={ob:2d}  n_ok={n_ok:3d}/{n_series_muestra:<3d} "
              "(excluidas por longitud: {n_excl:3d})  MASE mediano={mediana:.3f}  "
              "victoria vs naive={tasa:.1%}".format(
                  ob=r["outer_block"], n_ok=r["n_ok"], n_series_muestra=r["n_series_muestra"],
                  n_excl=r["n_excluidas_longitud_insuficiente"],
                  mediana=r["mase_ganador_mediana"], tasa=r["tasa_victoria_vs_naive"]))
    print("=" * 88)
    print("\nCSV resumen:", OUT_DIR / "sensibilidad_outer_block.csv")
    print("CSV detalle:", OUT_DIR / "sensibilidad_outer_block_detalle.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
