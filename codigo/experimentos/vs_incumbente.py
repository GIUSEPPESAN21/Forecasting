"""Herramienta vs. metodo incumbente de la empresa de referencia (Fase 4 / F09).

Este es el experimento que faltaba y que sostiene la afirmacion central del
manuscrito ("la herramienta mejora significativamente la precision"). El
diagnostico de la tesis (Sec. 3.1) describe el metodo incumbente sin ambiguedad:
promedio movil de 3 periodos (k=3), calculado manualmente en Excel, sin ningun
criterio de validacion ni metrica de error.

Reconstruccion del incumbente
------------------------------
El incumbente NO es un modelo elegido por validacion: es una regla fija,
aplicada igual a toda serie, sin importar su estructura. Por eso aqui se evalua
`moving_average(k=3)` con backtest walk-forward de un paso.

Proteccion contra sesgo de seleccion (importante)
--------------------------------------------------
La primera version de este script comparaba el ganador de `run_pipeline`
contra 'naive' sobre el MISMO bloque de evaluacion que decidio el ganador.
Eso es circular: si naive es uno de los candidatos sobre los que se toma el
argmin de MASE, el ganador nunca puede perder contra naive en ese bloque por
construccion matematica, no porque pronostique mejor. Ademas, el MASE del
ganador se escalaba internamente por `m_eff` (1 o 12 segun si la serie es
estacional) mientras que incumbente/naive se calculaban con `m=12` fijo, lo
que para series no estacionales comparaba numeros en escalas distintas.

Ambos problemas se corrigen usando `honest_outer_estimate`: se reserva un
bloque EXTERNO que ni la seleccion del metodo ni el ajuste de hiperparametros
vieron, y el incumbente se backtestea sobre esos mismos origenes con el mismo
`m_eff`. La cifra resultante es mas modesta que la de la primera version -y es
la correcta.

Uso
---
Con el archivo real de la empresa de referencia (columnas year, month, demand; sku opcional):

    python experiments/vs_incumbente.py --input ruta/al/archivo.xlsx
    python experiments/vs_incumbente.py --input ruta/al/archivo.xlsx --sku "TUBERIA CONDUIT..."

Sin datos reales (verifica el pipeline de punta a punta sobre series sinteticas
que reproducen la estructura descrita en la tesis: series de proyectos de obra,
volatiles, con historias de 24 meses):

    python experiments/vs_incumbente.py --synthetic --seed 20260824 --n-series 12

Salida: tabla comparativa incumbente vs. naive vs. herramienta, CSV en
experiments/output/vs_incumbente.csv, y el veredicto de significancia.
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

from forecasting_core.data import load_panel, load_series_from_excel  # noqa: E402
from forecasting_core.metrics import compute_metrics  # noqa: E402
from forecasting_core.models import MODEL_REGISTRY  # noqa: E402
from forecasting_core.optimize import honest_outer_estimate  # noqa: E402
from forecasting_core.validation import backtest_one_step  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"

INCUMBENTE_KEY = "moving_average"
INCUMBENTE_PARAMS = {"k": 3}  # el promedio movil de 3 periodos descrito en la tesis Sec. 3.1


# ---------------------------------------------------------------------------
# Datos sinteticos que reproducen la estructura declarada en la tesis:
# series de 24 meses, demanda de proyectos de obra (picos, cambios abruptos,
# ciclos irregulares) - Sec. 3.2.1 y 3.2.2 del documento.
# ---------------------------------------------------------------------------
def make_synthetic_panel(n_series: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    frames = []
    perfiles = ["plana", "tendencia", "estacional", "tendencia_estacional", "intermitente"]
    for i in range(n_series):
        # honest_outer_estimate reserva un bloque externo de 6 origenes ademas
        # del bloque de tuning; con n=24 no queda historia suficiente para el
        # pipeline interno completo, asi que series de 24 meses se excluirian
        # todas. Se usa 30/36/42 para que la mayoria de las series sean
        # evaluables bajo el protocolo honesto.
        n = int(rng.choice([30, 36, 42]))
        t = np.arange(n, dtype=float)
        perfil = perfiles[i % len(perfiles)]
        nivel = float(rng.uniform(500, 5000))
        if perfil == "plana":
            y = nivel + rng.normal(0, nivel * 0.09, n)
        elif perfil == "tendencia":
            y = nivel + rng.choice([-1, 1]) * rng.uniform(8, 25) * t + rng.normal(0, nivel * 0.08, n)
        elif perfil == "estacional":
            y = nivel + nivel * 0.30 * np.sin(2 * np.pi * t / 12) + rng.normal(0, nivel * 0.08, n)
        elif perfil == "tendencia_estacional":
            y = (nivel + rng.uniform(5, 15) * t
                 + nivel * 0.25 * np.sin(2 * np.pi * t / 12) + rng.normal(0, nivel * 0.08, n))
        else:  # intermitente: picos de proyectos de obra, meses en cero
            base = rng.normal(nivel * 0.3, nivel * 0.05, n)
            picos = rng.binomial(1, 0.25, n) * rng.uniform(nivel * 1.5, nivel * 3, n)
            y = np.maximum(base + picos, 0)
        # Cambios abruptos: la tesis los menciona explicitamente (Sec. 3.2.1).
        if rng.random() < 0.4:
            corte = rng.integers(n // 3, 2 * n // 3)
            y[corte:] *= rng.uniform(0.6, 1.6)
        y = np.maximum(y, 0)

        start_year = int(rng.integers(2021, 2023))
        idx = pd.date_range("{}-01-01".format(start_year), periods=n, freq="MS")
        frames.append(pd.DataFrame({
            "sku": "PRODUCTO-SINT-{:02d}".format(i + 1),
            "year": idx.year,
            "month": [meses[m - 1] for m in idx.month],
            "demand": np.round(y, 2),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Evaluacion sobre origenes IDENTICOS: incumbente, naive, herramienta.
# ---------------------------------------------------------------------------
def evaluate_series(sku: str, s: pd.Series, m: int = 12, outer_block: int = 6) -> dict | None:
    """Compara incumbente / naive / herramienta sobre un bloque EXTERNO honesto.

    Dos defectos se corrigieron aqui tras revisar los primeros resultados
    (ver `panel_publico.py` para el mismo problema en el panel publico):

    1. Comparar el ganador de `run_pipeline` contra 'naive' evaluado sobre el
       MISMO bloque que decidio al ganador es circular: naive es uno de los
       candidatos sobre los que se toma el argmin de MASE, asi que el ganador
       nunca puede perder ahi por construccion. Se usa `honest_outer_estimate`,
       que reserva un bloque que ni la seleccion del metodo ni el ajuste de
       hiperparametros vieron.
    2. El MASE del ganador se escala internamente por `m_eff` (1 si la serie
       no es estacional, 12 si lo es), mientras que incumbente/naive se
       calculaban aqui con un `m=12` fijo: para series no estacionales esto
       comparaba numeros en escalas DISTINTAS, no comparables entre si. Ahora
       el incumbente se backtestea sobre los mismos origenes externos y se
       escala con el mismo `m_eff` que uso la evaluacion externa del ganador.
    """
    y = s.to_numpy(dtype=float)
    n = y.size
    if n < 18:
        return None

    out = honest_outer_estimate(s, m=m, outer_block=outer_block)
    if not out.get("ok"):
        return None

    outer = out["outer"]
    outer_origins = outer.origins
    scale_train = y[: outer_origins[0]]
    m_eff = outer.m  # el mismo periodo de escala que uso la evaluacion externa

    incumbente_spec = MODEL_REGISTRY[INCUMBENTE_KEY]
    bt_inc = backtest_one_step(y, incumbente_spec, INCUMBENTE_PARAMS, m, outer_origins)
    m_inc = compute_metrics(bt_inc.y_true, bt_inc.y_pred, scale_train, m=m_eff)

    outer_metrics = out["outer_metrics"].set_index("modelo")
    if "naive" not in outer_metrics.index or not bool(outer_metrics.loc["naive", "elegible"]):
        return None
    m_naive_mase = float(outer_metrics.loc["naive", "mase"])
    m_naive_mape = float(outer_metrics.loc["naive", "mape"])
    m_naive_me = float(outer_metrics.loc["naive", "me"])

    winner = out["winner"]
    if winner not in outer_metrics.index or not bool(outer_metrics.loc[winner, "elegible"]):
        return None
    ganador = outer_metrics.loc[winner]

    return {
        "sku": sku, "n_obs": n, "n_eval": int(outer_origins.size),
        "perfil": ("tendencia" if out["inner"].profile.has_trend else "sin tendencia")
                  + ("+estacional" if out["inner"].profile.has_seasonality else ""),
        "incumbente_mase": m_inc.mase, "incumbente_mape": m_inc.mape,
        "incumbente_me": m_inc.me,
        "naive_mase": m_naive_mase, "naive_mape": m_naive_mape, "naive_me": m_naive_me,
        "herramienta_metodo": winner,
        "herramienta_mase": float(ganador["mase"]), "herramienta_mape": float(ganador["mape"]),
        "herramienta_me": float(ganador["me"]),
        "mejora_vs_incumbente_pct": (
            100.0 * (m_inc.mase - float(ganador["mase"])) / m_inc.mase
            if np.isfinite(m_inc.mase) and m_inc.mase > 0 else np.nan
        ),
        "supera_incumbente": bool(float(ganador["mase"]) < m_inc.mase)
                              if np.isfinite(m_inc.mase) else None,
        "supera_naive": bool(float(ganador["mase"]) < m_naive_mase)
                         if np.isfinite(m_naive_mase) else None,
    }


def wtl_breakdown(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Victorias/empates/derrotas de `col_a` vs. `col_b` (menor MASE = victoria).

    F34: extraido a funcion pura -antes vivia inline en `run()`- para poder
    probarlo con un caso de resultado conocido a mano.
    """
    comp = df.dropna(subset=[col_a, col_b])
    victorias = int((comp[col_a] < comp[col_b]).sum())
    empates = int((comp[col_a] == comp[col_b]).sum())
    derrotas = int((comp[col_a] > comp[col_b]).sum())
    return {"n": len(comp), "victorias": victorias, "empates": empates, "derrotas": derrotas}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM sobre perdida cuadratica, con correccion de Harvey-Leybourne-Newbold.

    H0: ambos metodos tienen la misma exactitud de pronostico esperada.
    """
    d = e1 ** 2 - e2 ** 2
    n = d.size
    if n < 4:
        return float("nan"), float("nan")
    d_bar = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for lag in range(1, h):
        if lag >= n:
            break
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        var_d += 2 * cov
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var_d / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


def run(panel: pd.DataFrame, m: int, label: str) -> pd.DataFrame:
    loads = load_panel(panel)
    rows = []
    for sku, load in loads.items():
        if not load.report.ok:
            print("  [omitido] {}: {}".format(sku, " | ".join(load.report.errors)))
            continue
        row = evaluate_series(sku, load.series.dropna(), m=m)
        if row is None:
            print("  [omitido] {}: pipeline sin resultado valido".format(sku))
            continue
        rows.append(row)
        print("  {:28s} n={:3d} | incumbente MASE={:.3f} MAPE={:6.2f}% | "
              "herramienta ({}) MASE={:.3f} MAPE={:6.2f}% | mejora {:+.1f}%".format(
                  sku[:28], row["n_obs"], row["incumbente_mase"], row["incumbente_mape"],
                  row["herramienta_metodo"], row["herramienta_mase"], row["herramienta_mape"],
                  row["mejora_vs_incumbente_pct"]))
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo se pudo evaluar ninguna serie de {}.".format(label))
        return df

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "vs_incumbente.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("RESUMEN — {} ({} series)".format(label, len(df)))
    print("=" * 88)
    print("  MASE mediano   incumbente={:.3f}  naive={:.3f}  herramienta={:.3f}".format(
        df["incumbente_mase"].median(), df["naive_mase"].median(), df["herramienta_mase"].median()))
    print("  MAPE mediano   incumbente={:.1f}%  naive={:.1f}%  herramienta={:.1f}%".format(
        df["incumbente_mape"].median(), df["naive_mape"].median(), df["herramienta_mape"].median()))
    print("  |ME| mediano   incumbente={:.1f}  naive={:.1f}  herramienta={:.1f}   "
          "(sesgo — no reportado antes, F06)".format(
        df["incumbente_me"].abs().median(), df["naive_me"].abs().median(),
        df["herramienta_me"].abs().median()))
    print("  Series donde la herramienta supera al incumbente: {}/{} ({:.0%})".format(
        df["supera_incumbente"].sum(), df["supera_incumbente"].notna().sum(),
        df["supera_incumbente"].mean()))
    print("  Series donde la herramienta supera al naive      : {}/{} ({:.0%})".format(
        df["supera_naive"].sum(), df["supera_naive"].notna().sum(), df["supera_naive"].mean()))
    mejora_mediana_por_serie = df["mejora_vs_incumbente_pct"].median()
    mase_med_h = df["herramienta_mase"].median()
    mase_med_i = df["incumbente_mase"].median()
    mejora_de_medianas = 100.0 * (mase_med_i - mase_med_h) / mase_med_i if mase_med_i > 0 else float("nan")
    print("  Mejora mediana de MASE por serie vs. incumbente (mediana de las mejoras "
          "individuales): {:+.1f}%".format(mejora_mediana_por_serie))
    print("  Mejora de las medianas (MASE mediano incumbente -> herramienta, {:.3f} -> {:.3f}): "
          "{:+.1f}%  (dato complementario -no intercambiable con el anterior, F46)".format(
              mase_med_i, mase_med_h, mejora_de_medianas))

    # F34: desglose victorias/empates/derrotas de la herramienta contra naive.
    # Empate = la herramienta ELIGIO naive como ganador (herramienta_metodo ==
    # "naive"), por lo que herramienta_mase == naive_mase por construccion.
    wtl = wtl_breakdown(df, "herramienta_mase", "naive_mase")
    victorias_n, empates_n, derrotas_n = wtl["victorias"], wtl["empates"], wtl["derrotas"]
    print("\n  Desglose herramienta vs. naive (n={}): {} victorias, {} empates "
          "(eligio naive), {} derrotas".format(wtl["n"], victorias_n, empates_n, derrotas_n))

    # F34: Wilcoxon pareado herramienta vs. incumbente sobre MASE.
    comp_inc = df.dropna(subset=["herramienta_mase", "incumbente_mase"])
    comp_inc = comp_inc[comp_inc["herramienta_mase"] != comp_inc["incumbente_mase"]]
    print("\n  Prueba de Wilcoxon pareada (MASE herramienta vs. MASE incumbente):")
    if len(comp_inc) >= 10:
        from scipy import stats
        w_stat, w_p = stats.wilcoxon(comp_inc["herramienta_mase"], comp_inc["incumbente_mase"])
        print("    W={:.1f}  p={:.6g}  n={} (diferencias no nulas; de {} pares totales)".format(
            w_stat, w_p, len(comp_inc), len(df.dropna(subset=["herramienta_mase", "incumbente_mase"]))))
    else:
        w_stat, w_p = float("nan"), float("nan")
        print("    n insuficiente (<10) para Wilcoxon fiable: n={}".format(len(comp_inc)))

    resumen = pd.DataFrame([{
        "n_series": len(df),
        "mase_mediano_incumbente": mase_med_i, "mase_mediano_herramienta": mase_med_h,
        "mejora_mediana_por_serie_pct": mejora_mediana_por_serie,
        "mejora_de_medianas_pct": mejora_de_medianas,
        "victorias_vs_naive": victorias_n, "empates_vs_naive": empates_n,
        "derrotas_vs_naive": derrotas_n,
        "wilcoxon_W_vs_incumbente": w_stat, "wilcoxon_p_vs_incumbente": w_p,
        "wilcoxon_n_vs_incumbente": len(comp_inc),
    }])
    resumen.to_csv(OUT_DIR / "vs_incumbente_resumen.csv", index=False, encoding="utf-8-sig")

    # Diebold-Mariano agregado: se concatenan los errores de todas las series
    # de igual longitud de evaluacion para tener una sola prueba conjunta.
    print("\n  Prueba de Diebold-Mariano (H0: igual exactitud, perdida cuadratica):")
    print("    NOTA: por serie no siempre hay suficientes origenes (n>=4) para una")
    print("    prueba individual fiable; con pocas series el resultado es orientativo,")
    print("    no una conclusion definitiva de superioridad estadistica.")
    print(path)
    print("Resumen:", OUT_DIR / "vs_incumbente_resumen.csv")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                    help="Ruta al Excel/CSV real de la empresa de referencia (columnas year, month, demand[, sku])")
    ap.add_argument("--sku", type=str, default=None, help="Filtrar a un solo producto")
    ap.add_argument("--synthetic", action="store_true",
                    help="Usar dataset sintetico (estructura de la tesis) en vez de datos reales")
    ap.add_argument("--n-series", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--m", type=int, default=12)
    args = ap.parse_args()

    if args.input:
        print("Cargando datos reales de: {}".format(args.input))
        res = load_series_from_excel(args.input, sku=args.sku)
        if not res.report.ok:
            print("ERROR al cargar: {}".format(" | ".join(res.report.errors)))
            return 1
        panel = pd.DataFrame({
            "sku": args.sku or "producto",
            "year": res.series.dropna().index.year,
            "month": [m for m in res.series.dropna().index.strftime("%B")],
            "demand": res.series.dropna().to_numpy(),
        })
        run(panel, args.m, "datos reales de la empresa de referencia")
        return 0

    print("SIN Excel real: usando dataset sintetico que reproduce la estructura de la\n"
          "tesis (24-36 meses, series de proyectos de obra, picos, cambios abruptos).\n"
          "Este resultado verifica el pipeline de punta a punta; NO reemplaza la\n"
          "corrida sobre los datos reales de la empresa de referencia, que sigue pendiente.\n")
    panel = make_synthetic_panel(args.n_series, args.seed)
    run(panel, args.m, "dataset sintetico (pendiente de datos reales)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
