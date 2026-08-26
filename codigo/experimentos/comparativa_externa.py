"""Herramienta vs. Prophet vs. LightGBM, bajo UN SOLO protocolo (Fase 11, F26/F27).

Origen de este script
----------------------
Los tutores del proyecto pidieron una comparacion de la Herramienta contra
Prophet y "otro pronosticador famoso" (aqui: LightGBM, ganador historico de la
competencia M5, usado como modelo GLOBAL via `mlforecast`). Adjuntaron un PDF
(`comparacion_herramientas.pdf`) con 10 corridas manuales de "Herramienta vs.
Prophet" sobre series de 24 a 180 meses.

Ese PDF no es reutilizable como evidencia (hallazgo F26, documentado en
`docs/prompt_maestro.md` de la Fase 11): mezcla el MAPE walk-forward interno
de la Herramienta con las metricas de Prophet calculadas sobre un holdout
aparte, con `yearly_seasonality` de Prophet activada de forma incondicional
incluso en la serie de 24 observaciones (2 ciclos anuales exactos, el minimo
teorico) -de ahi el MAPE=353.98% y los pronosticos negativos que reporta ese
documento para n=24. Comparar un numero calculado dentro de la seleccion del
modelo contra un numero calculado fuera de ella no es una comparacion valida;
es la misma clase de sesgo de circularidad que `honest_outer_estimate()` ya
corrige para la Herramienta (ver `RESUMEN_EJECUCION.md`, "Correccion post-hoc").

Protocolo aqui (unico, honesto, identico para los tres metodos)
------------------------------------------------------------------
1. La Herramienta se evalua con `honest_outer_estimate()`: bloque de tuning,
   bloque de evaluacion (elige el metodo) y bloque EXTERNO reservado, que ni
   la eleccion de hiperparametros ni la eleccion del metodo vieron.
2. Prophet y LightGBM NO tienen hiperparametros que ajustar aqui (instruccion
   explicita: sin tuning agresivo, son lineas base de comparacion). Se
   evaluan por backtest de un paso directamente sobre el MISMO bloque EXTERNO
   que uso la Herramienta -exactamente el mismo patron que ya usa
   `vs_incumbente.py` para el metodo incumbente de Tuboplex (que tampoco se
   tunea sobre ese bloque).
3. Las cuatro metricas (MASE, MAPE, MAD, MSE, ME) se calculan con
   `forecasting_core.metrics.compute_metrics`, con el MISMO `scale_train` y
   `m_eff` para los tres metodos, para que el MASE sea comparable entre ellos.

`honest_outer_estimate()` exige `n - outer_block >= 22` internamente (piso de
entrenamiento + tuning + evaluacion del pipeline interno). Para incluir
n=24 -la longitud exacta que rompio a Prophet en el PDF de los tutores- este
script reduce `outer_block` a 2 solo en ese caso (`choose_outer_block`),
declarado explicitamente en el CSV (`outer_block`) y en el log: es MENOS
origenes de los que se usarian con mas historia, no un area gris oculta.

Regimenes estructurales evaluados por cada longitud de serie (n=24, 36, 48,
60, 72, 84, 96, 120, 150, 180 -el mismo barrido del PDF de los tutores):
plano, tendencia, estacional, tendencia+estacional, y corta/erratica (picos
de proyectos de obra con meses en cero, el patron que documenta la tesis en
su Sec. 3.2.1 y que es el regimen n=24 que aparece en el PDF adjunto).

Prophet y LightGBM son OPCIONALES: si alguno de los dos paquetes no esta
instalado en el entorno de ejecucion, este script sigue corriendo con el que
si lo este y lo declara en el log (ver `codigo/experimentos/decision_prophet.md`,
seccion "Actualizacion Fase 11", para el estado de instalacion documentado en
este repositorio). Nunca se bloquea la fase completa por un solo paquete.

Uso
---
    python codigo/experimentos/comparativa_externa.py --seed 20260824
    python codigo/experimentos/comparativa_externa.py --lengths 24 48 96 --n-per-cell 2

Salida: `resultados/comparativa_externa.csv` (una fila por serie evaluada).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from external_baselines.adapters import external_specs  # noqa: E402
from forecasting_core.metrics import compute_metrics  # noqa: E402
from forecasting_core.optimize import honest_outer_estimate  # noqa: E402
from forecasting_core.validation import backtest_one_step  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("comparativa_externa")

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"

DEFAULT_LENGTHS = (24, 36, 48, 60, 72, 84, 96, 120, 150, 180)
REGIMES = ("plano", "tendencia", "estacional", "tendencia_estacional", "corta_erratica")

# Piso interno de `honest_outer_estimate`: n - outer_block >= 22 (ver optimize.py:
# ABSOLUTE_MIN_TRAIN=10 + MIN_TUNE_ORIGINS=4 + MIN_ORIGINS=8). No se toca ese
# modulo (restriccion de aislamiento de la Fase 11); en vez de eso este script
# reduce `outer_block` solo para las longitudes mas cortas del barrido.
_INNER_FLOOR = 22
DEFAULT_OUTER_BLOCK = 6
MIN_OUTER_BLOCK = 2


def choose_outer_block(n: int) -> int | None:
    """`outer_block` mas grande posible sin violar el piso interno del pipeline.

    n=24 con el `outer_block=6` por defecto de `vs_incumbente.py`/`panel_publico.py`
    fallaria (24-6=18 < 22); aqui se reduce a 2 -declarado en el CSV como
    `outer_block`, no oculto- para poder incluir exactamente la longitud que
    rompio a Prophet en el PDF de los tutores. Devuelve None si ni el minimo
    (2) cabe.
    """
    ob = min(DEFAULT_OUTER_BLOCK, n - _INNER_FLOOR)
    if ob < MIN_OUTER_BLOCK:
        return None
    return int(ob)


# ---------------------------------------------------------------------------
# Panel sintetico: 10 longitudes x 5 regimenes estructurales (F27, F28).
# Mismo estilo que `vs_incumbente.py::make_synthetic_panel` (proyectos de
# obra: picos, cambios abruptos, ciclos irregulares - tesis Sec. 3.2.1/3.2.2),
# pero con n y regimen fijados EXPLICITAMENTE por celda en vez de aleatorios,
# para que el barrido cubra exactamente lo que pide el prompt de la Fase 11.
# ---------------------------------------------------------------------------
def make_regime_series(n: int, regimen: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    nivel = float(rng.uniform(500, 5000))

    if regimen == "plano":
        y = nivel + rng.normal(0, nivel * 0.08, n)
    elif regimen == "tendencia":
        y = nivel + rng.choice([-1.0, 1.0]) * rng.uniform(6, 20) * t + rng.normal(0, nivel * 0.07, n)
    elif regimen == "estacional":
        y = nivel + nivel * 0.30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, nivel * 0.07, n)
    elif regimen == "tendencia_estacional":
        y = (nivel + rng.uniform(4, 12) * t
             + nivel * 0.25 * np.sin(2 * np.pi * t / 12) + rng.normal(0, nivel * 0.07, n))
    elif regimen == "corta_erratica":
        # Picos de proyectos de obra con meses en cero (tesis Sec. 3.2.1): el
        # patron que en el PDF de los tutores, con n=24 y Prophet mal
        # configurado, produjo MAPE=353.98% y pronosticos negativos (F26).
        base = rng.normal(nivel * 0.30, nivel * 0.06, n)
        picos = rng.binomial(1, 0.25, n) * rng.uniform(nivel * 1.5, nivel * 3.0, n)
        y = base + picos
    else:
        raise ValueError("regimen desconocido: {}".format(regimen))

    y = np.maximum(y, 0.0)
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    return pd.Series(np.round(y, 2), index=idx, name="demand")


# ---------------------------------------------------------------------------
# Evaluacion de una celda (n, regimen): Herramienta vs. Prophet vs. LightGBM,
# los tres sobre el MISMO bloque externo.
# ---------------------------------------------------------------------------
def _metric_cols(prefix: str, ms) -> dict:
    return {
        "{}_mase".format(prefix): ms.mase,
        "{}_mape".format(prefix): ms.mape,
        "{}_mad".format(prefix): ms.mad,
        "{}_mse".format(prefix): ms.mse,
        "{}_me".format(prefix): ms.me,
    }


def evaluate_cell(n: int, regimen: str, seed: int, specs: dict, *, m: int = 12) -> dict:
    base = {"n": n, "regimen": regimen, "seed": seed}
    outer_block = choose_outer_block(n)
    if outer_block is None:
        return dict(base, estado="n_insuficiente",
                    detalle="n={} no alcanza el piso interno del pipeline (n-outer_block>={})".format(
                        n, _INNER_FLOOR))

    s = make_regime_series(n, regimen, seed)
    out = honest_outer_estimate(s, m=m, outer_block=outer_block)
    if not out.get("ok"):
        return dict(base, estado="sin_ganador", outer_block=outer_block,
                    detalle=out.get("reason", ""))

    outer = out["outer"]
    origins = outer.origins
    y = s.to_numpy(dtype=float)
    scale_train = y[: origins[0]]
    m_eff = outer.m
    winner = out["winner"]

    row = dict(base, estado="ok", outer_block=outer_block, n_eval=int(origins.size),
               herramienta_metodo=winner)

    outer_metrics = out["outer_metrics"].set_index("modelo")
    for metodo, prefix in (("naive", "naive"), ("seasonal_naive", "seasonal_naive"),
                            (winner, "herramienta")):
        if metodo in outer_metrics.index and bool(outer_metrics.loc[metodo, "elegible"]):
            r = outer_metrics.loc[metodo]
            row.update({
                "{}_mase".format(prefix): float(r["mase"]),
                "{}_mape".format(prefix): float(r["mape"]),
                "{}_mad".format(prefix): float(r["mad"]),
                "{}_mse".format(prefix): float(r["mse"]),
                "{}_me".format(prefix): float(r["me"]),
            })

    # Prophet / LightGBM: backtest directo sobre el bloque EXTERNO, sin
    # tuning (mismo patron que el incumbente en `vs_incumbente.py`).
    for key, spec in specs.items():
        short = key.replace("ext_", "")
        try:
            bt = backtest_one_step(y, spec, None, m, origins)
        except Exception as exc:  # nunca tumbar el panel completo por una serie
            row["{}_estado".format(short)] = "error: {}: {}".format(type(exc).__name__, exc)
            logger.warning("%s fallo en n=%d regimen=%s: %s", key, n, regimen, exc)
            continue
        if not bt.complete:
            reason = bt.failures[0] if bt.failures else "origenes incompletos"
            row["{}_estado".format(short)] = "incompleto: {}".format(reason)
            continue
        ms = compute_metrics(bt.y_true, bt.y_pred, scale_train, m=m_eff)
        row["{}_estado".format(short)] = "ok"
        row.update(_metric_cols(short, ms))

    return row


def run(lengths, regimenes, n_per_cell: int, seed: int, m: int) -> pd.DataFrame:
    specs = external_specs()
    if not specs:
        print("ATENCION: ni prophet ni mlforecast/lightgbm estan instalados. "
              "Instale requirements-external.txt para obtener comparadores externos; "
              "este panel solo reportara Herramienta vs. naive/seasonal_naive. "
              "Ver codigo/experimentos/decision_prophet.md.")
    else:
        print("Comparadores externos disponibles: {}".format(", ".join(sorted(specs))))
        faltantes = {"ext_prophet", "ext_lightgbm"} - set(specs)
        if faltantes:
            print("  (no disponibles en este entorno: {} -no bloquea la fase, F27 seccion 3)".format(
                ", ".join(sorted(faltantes))))

    rows = []
    t0 = time.perf_counter()
    total = len(lengths) * len(regimenes) * n_per_cell
    done = 0
    for n in lengths:
        for regimen in regimenes:
            for rep in range(n_per_cell):
                cell_seed = seed + 1000 * n + 17 * hash(regimen) % 9973 + rep
                cell_seed = abs(cell_seed) % (2**31 - 1)
                row = evaluate_cell(n, regimen, cell_seed, specs, m=m)
                row["rep"] = rep
                rows.append(row)
                done += 1
                estado = row.get("estado")
                print("  [{:3d}/{:3d}] n={:3d} regimen={:22s} -> {}".format(
                    done, total, n, regimen, estado))

    df = pd.DataFrame(rows)
    elapsed = time.perf_counter() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "comparativa_externa.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    ok = df[df["estado"] == "ok"].copy()
    n_fail = len(df) - len(ok)
    print("\n" + "=" * 92)
    print("COMPARATIVA EXTERNA -- {} celdas ({} ok, {} sin resultado valido) -- {:.1f}s".format(
        len(df), len(ok), n_fail, elapsed))
    print("=" * 92)
    if n_fail:
        print("Motivos de exclusion (sin cobertura silenciosa):")
        for motivo, cnt in df.loc[df["estado"] != "ok", "estado"].value_counts().items():
            print("  {}: {}".format(motivo, cnt))

    if len(ok):
        metodos = ["herramienta", "naive", "seasonal_naive"]
        if "ext_prophet" in specs:
            metodos.append("prophet")
        if "ext_lightgbm" in specs:
            metodos.append("lightgbm")

        print("\nMASE mediano por metodo y por regimen estructural:")
        cols = ["{}_mase".format(m_) for m_ in metodos if "{}_mase".format(m_) in ok.columns]
        resumen = ok.groupby("regimen")[cols].median().round(3)
        print(resumen.to_string())

        print("\nMASE mediano por metodo y por longitud de serie:")
        resumen_n = ok.groupby("n")[cols].median().round(3)
        print(resumen_n.to_string())

        print("\nMASE mediano por metodo (global, {} series):".format(len(ok)))
        for col in cols:
            v = ok[col].dropna()
            if len(v):
                print("  {:26s} mediana={:.3f}  media={:.3f}  n={}".format(col, v.median(), v.mean(), len(v)))

        for ext in ("prophet", "lightgbm"):
            hcol, ecol = "herramienta_mase", "{}_mase".format(ext)
            if ecol in ok.columns:
                comp = ok.dropna(subset=[hcol, ecol])
                if len(comp):
                    supera = (comp[hcol] < comp[ecol]).mean()
                    print("\nLa Herramienta supera a {} en {:.0%} de las {} series comparables "
                          "(MASE menor = mejor).".format(ext, supera, len(comp)))

    print("\nCSV:", path)
    print("Tiempo total: {:.1f}s ({} celdas evaluadas)".format(elapsed, len(df)))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    ap.add_argument("--regimes", type=str, nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--n-per-cell", type=int, default=1,
                    help="Series sinteticas por combinacion (n, regimen). Default 1 "
                         "(= 1 serie por celda del barrido, como en el PDF de los tutores).")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--m", type=int, default=12)
    # --input existe por consistencia con vs_incumbente.py, pero este script
    # evalua Prophet/LightGBM sobre TODAS las longitudes del barrido: con un
    # solo archivo real no hay forma de variar n de forma controlada, asi que
    # se documenta la limitacion en vez de fingir soporte que no tiene sentido.
    ap.add_argument("--input", type=str, default=None,
                    help="No soportado en este script (requiere variar n de forma controlada); "
                         "se ignora si se pasa. Use vs_incumbente.py --input para datos reales.")
    args = ap.parse_args()
    if args.input:
        print("AVISO: --input no esta soportado en comparativa_externa.py (requiere variar n "
              "de forma controlada por celda). Se ignora y se corre el panel sintetico.")

    run(args.lengths, args.regimes, args.n_per_cell, args.seed, args.m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
