"""Validacion Monte Carlo de la clasificacion estructural (Fase 8.1).

Prueba de aceptacion de la Fase 1: las tasas de falso positivo de los tests de
tendencia y estacionalidad deben caer del 50-74% medido en el codigo original a
valores cercanos al nivel nominal del 5%.

Uso
---
    python experiments/montecarlo_clasificacion.py --reps 1000 --seed 20260824
    python experiments/montecarlo_clasificacion.py --reps 200 --quick

Salida: tabla en consola + CSV en experiments/output/montecarlo_clasificacion.csv
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

from forecasting_core.classification import classify_series  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"


# ---------------------------------------------------------------------------
# Generadores. Cada uno declara cual es la VERDAD sobre tendencia/estacionalidad
# ---------------------------------------------------------------------------
def make_generators(rng: np.random.Generator):
    def ruido(n):
        return 1000.0 + rng.normal(0, 100, n)

    def ar07(n):
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + rng.normal(0, 100)
        return 1000.0 + y

    def random_walk(n):
        return 1000.0 + np.cumsum(rng.normal(0, 60, n))

    def tendencia(n):
        return 1000.0 + 25.0 * np.arange(n) + rng.normal(0, 100, n)

    def rw_drift(n):
        return 1000.0 + np.cumsum(20.0 + rng.normal(0, 60, n))

    def solo_tendencia_limpia(n):
        return 1000.0 + 25.0 * np.arange(n) + rng.normal(0, 40, n)

    def estacional(n):
        return 1000.0 + 350.0 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 120, n)

    def tend_estacional(n):
        return (1000.0 + 20.0 * np.arange(n)
                + 350.0 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 120, n))

    # (nombre, generador, tendencia_verdadera, estacionalidad_verdadera)
    return [
        ("ruido blanco",        ruido,                  False, False),
        ("AR(1) phi=0.7",       ar07,                   False, False),
        ("paseo aleatorio",     random_walk,            False, False),
        ("tendencia lineal",    tendencia,              True,  False),
        ("tendencia limpia",    solo_tendencia_limpia,  True,  False),
        ("paseo con deriva",    rw_drift,               True,  False),
        ("estacional pura",     estacional,             False, True),
        ("tendencia+estacion.", tend_estacional,        True,  True),
    ]


def run(reps: int, sizes: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gens = make_generators(rng)
    rows = []
    total = len(gens) * len(sizes)
    done = 0
    for name, gen, true_trend, true_seas in gens:
        for n in sizes:
            t0 = time.perf_counter()
            hits_trend = hits_seas = n_no_eval = 0
            fs_vals = []
            for _ in range(reps):
                # Se valida `classify_series`, que es lo que ejecuta el usuario:
                # incluye el orden estacionalidad -> desestacionalizar -> tendencia.
                prof = classify_series(gen(n), m=12)
                hits_trend += int(prof.has_trend == true_trend)
                if not np.isfinite(prof.seasonal_strength):
                    n_no_eval += 1
                else:
                    hits_seas += int(prof.has_seasonality == true_seas)
                fs_vals.append(prof.seasonal_strength)
            n_seas_eval = reps - n_no_eval
            rate_trend = hits_trend / reps
            rate_seas = (hits_seas / n_seas_eval) if n_seas_eval else float("nan")
            rows.append({
                "serie": name, "n": n, "reps": reps,
                "tendencia_verdadera": true_trend,
                "acierto_tendencia": rate_trend,
                "error_tendencia": 1 - rate_trend,
                "tipo_error_tendencia": "falso positivo" if not true_trend else "falso negativo",
                # F37: eje de analisis explicito por columna, sin depender de que
                # quien consuma el CSV infiera "tamano vs. potencia" a partir del
                # booleano de verdad. tamano = tasa de falso positivo (H0 cierta,
                # se mide el nivel del test); potencia = tasa de verdadero
                # positivo (H1 cierta, se mide la capacidad de detectarla).
                "tipo_tendencia": "tamano" if not true_trend else "potencia",
                "estacionalidad_verdadera": true_seas,
                "estacionalidad_no_evaluable": n_no_eval / reps,
                "acierto_estacionalidad": rate_seas,
                "error_estacionalidad": (1 - rate_seas) if n_seas_eval else float("nan"),
                "tipo_error_estacionalidad": "falso positivo" if not true_seas else "falso negativo",
                "tipo_estacionalidad": "tamano" if not true_seas else "potencia",
                "F_S_mediana": float(np.nanmedian(fs_vals)) if n_seas_eval else float("nan"),
                "segundos": round(time.perf_counter() - t0, 2),
            })
            done += 1
            seas_txt = ("no evaluable" if not n_seas_eval
                        else "{:5.1%} correcto".format(rate_seas))
            print("  [{}/{}] {:22s} n={:<4d} tendencia {:5.1%} correcto | "
                  "estacionalidad {}".format(
                      done, total, name, n, rate_trend, seas_txt), flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("TASAS DE ERROR DE LA CLASIFICACION (objetivo: ~5% donde la verdad es 'No')")
    print("=" * 84)
    for label, col_true, col_err, col_kind in [
        ("TENDENCIA", "tendencia_verdadera", "error_tendencia", "tipo_error_tendencia"),
        ("ESTACIONALIDAD", "estacionalidad_verdadera", "error_estacionalidad",
         "tipo_error_estacionalidad"),
    ]:
        print("\n--- {} ---".format(label))
        piv = df.pivot_table(index=["serie", col_true], columns="n", values=col_err)
        for (serie, truth), row in piv.iterrows():
            kind = "falso POSITIVO" if not truth else "falso negativo"
            cells = "".join(
                "       n/e" if not np.isfinite(v) else "{:>10.1%}".format(v)
                for v in row.values
            )
            print("  {:22s} verdad={:5s} {:15s}{}".format(
                serie, "Si" if truth else "No", kind, cells))
        print("  {:22s} {:21s}{}".format(
            "", "n =", "".join("{:>10d}".format(c) for c in piv.columns)))
    print("\n  n/e = no evaluable: la serie no tiene los 3 ciclos completos que exige")
    print("        la identificacion de estacionalidad. Es una respuesta valida, no un error.")

    fp_trend = df[~df["tendencia_verdadera"]]["error_tendencia"].dropna()
    fp_seas = df[~df["estacionalidad_verdadera"]]["error_estacionalidad"].dropna()
    pw_seas = 1 - df[df["estacionalidad_verdadera"]]["error_estacionalidad"].dropna()
    print("\n" + "-" * 84)
    print("RESUMEN")
    print("  Falsos positivos de TENDENCIA      : media {:.1%}  maximo {:.1%}   "
          "[codigo original: 74.4%]".format(fp_trend.mean(), fp_trend.max()))
    print("  Falsos positivos de ESTACIONALIDAD : media {:.1%}  maximo {:.1%}   "
          "[codigo original: 50.2%]".format(fp_seas.mean(), fp_seas.max()))
    if len(pw_seas):
        print("  Potencia de ESTACIONALIDAD (donde es evaluable): media {:.1%}".format(
            pw_seas.mean()))
    ok = fp_trend.max() <= 0.20 and fp_seas.max() <= 0.10
    print("\n  ACEPTACION FASE 1: {}".format(
        "PASA" if ok else "NO PASA"))
    print("    criterio: falsos positivos de tendencia <= 20% (limitado por la")
    print("              distorsion de tamano del ADF con n=24) y de estacionalidad <= 10%.")
    print("-" * 84)

    # F37: potencia del test de tendencia por escenario y tamano muestral -
    # el eje que quedaba sin reportar (solo se documentaba el tamano/FP).
    print("\n" + "=" * 84)
    print("POTENCIA DEL TEST DE TENDENCIA (tipo_tendencia == 'potencia', H1 cierta)")
    print("=" * 84)
    pot = df[df["tipo_tendencia"] == "potencia"]
    if len(pot):
        piv_pot = pot.pivot_table(index="serie", columns="n", values="acierto_tendencia")
        for serie, row in piv_pot.iterrows():
            cells = "".join(
                "       n/e" if not np.isfinite(v) else "{:>10.1%}".format(v)
                for v in row.values
            )
            print("  {:22s}{}".format(serie, cells))
        print("  {:22s}{}".format("n =", "".join("{:>10d}".format(c) for c in piv_pot.columns)))
        print("\n  NOTA: es justamente en el regimen objetivo de este trabajo (series cortas,")
        print("        n<=36) donde la potencia del test de tendencia es mas debil -ver la")
        print("        columna n=24/n=36 arriba frente a n=120.")
    else:
        print("  (sin escenarios de potencia en esta corrida; incluya --sizes con series de")
        print("   tendencia verdadera para reportarla)")
    print("-" * 84)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 120])
    ap.add_argument("--quick", action="store_true", help="200 replicas, n=24 y 48")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.sizes = 200, [24, 48]

    print("Monte Carlo de clasificacion: {} replicas, n={}, semilla={}".format(
        args.reps, args.sizes, args.seed))
    df = run(args.reps, args.sizes, args.seed)
    report(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "montecarlo_clasificacion.csv"
    df.to_csv(path, index=False)
    print("\nCSV: {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
