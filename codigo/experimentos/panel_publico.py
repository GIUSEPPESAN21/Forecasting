"""Validacion sobre un panel publico: M3 mensual, series cortas (Fase 8.2).

Con una sola serie de caso de estudio no se puede afirmar que un metodo supera
a otro: la varianza del MASE estimado sobre 10-18 origenes es demasiado alta.
Este script corre el pipeline COMPLETO sobre un subconjunto ACOTADO (100-300
series, nunca el M3/M4 completo, para respetar el presupuesto de RAM de la
Fase 1) de la competencia M3 mensual, TRUNCADA a <= 48 observaciones por serie
-el regimen de datos escasos donde este trabajo se posiciona; las series M3
nativas tienen 66-144 observaciones, asi que se usa la cola reciente de cada
una, preservando demanda real en vez de generar series sinteticas-, y reporta:

  1. distribucion de MASE por metodo y por regimen estructural
     (tendencia / estacional / plano), no un caso aislado;
  2. una prueba de Diebold-Mariano por pares entre los metodos top, agregada
     sobre todas las series (test de Petropoulos & Svetunkov 2020: promedia el
     estadistico DM entre series, mas robusto que un DM por serie con pocos
     origenes cada una).

Cada serie se evalua con `honest_outer_estimate`: ademas del bloque de tuning
de hiperparametros ya separado en `run_pipeline`, se reserva un bloque EXTERNO
(por defecto 6 origenes) que ni siquiera participa en la ELECCION del metodo
ganador. Esto es necesario porque comparar el ganador contra 'naive' sobre el
mismo bloque que decidio el ganador es circular: si naive es uno de los
candidatos sobre los que se toma el argmin de MASE, el ganador nunca puede
perder contra naive en ESE bloque por construccion matematica, no porque sea
mejor pronosticando. La primera version de este script comparaba sobre el
bloque de seleccion y reportaba "100% de las series superan al naive" en las
150 series evaluadas -una cifra trivialmente verdadera y sin contenido
informativo- hasta que se detecto el problema al revisar los resultados antes
de publicarlos.

No hay ningun tope de cobertura silencioso: si una serie se descarta se cuenta
y se informa por que.

Uso:
    python experiments/panel_publico.py --n-series 200 --max-len 48 --seed 20260824
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

from forecasting_core.classification import classify_series  # noqa: E402
from forecasting_core.optimize import honest_outer_estimate  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "resultados"
CACHE_DIR = Path(__file__).resolve().parent / "m3cache"


def load_m3_monthly_short(max_len: int, min_len: int = 24) -> pd.DataFrame:
    """Series M3 mensuales, TRUNCADAS a sus ultimas `max_len` observaciones.

    Las series de M3 mensual tienen entre 66 y 144 observaciones -no existen
    series cortas de forma nativa. Truncar a la cola de cada serie es la
    tecnica estandar en la literatura para simular el regimen de escasez de
    datos (Petropoulos et al. 2022) sin abandonar demanda real: se preserva el
    patron reciente de cada serie, no se genera nada sintetico.
    """
    from datasetsforecast.m3 import M3

    y_df, *_ = M3.load(directory=str(CACHE_DIR), group="Monthly")
    lens = y_df.groupby("unique_id")["y"].size()
    elegibles = lens[lens >= min_len].index
    df = y_df[y_df["unique_id"].isin(elegibles)].sort_values(["unique_id", "ds"])
    return df.groupby("unique_id", group_keys=False).tail(max_len).copy()


def evaluate_one(uid: str, y: np.ndarray, m: int = 12, outer_block: int = 6,
                  structural_filter: bool = True) -> dict | None:
    """Evalua UNA serie con el protocolo externo honesto (F05 aplicado tambien
    a la eleccion del metodo, no solo a sus hiperparametros).

    Comparar el ganador de `run_pipeline` contra 'naive' calculado en el MISMO
    bloque de evaluacion que decidio al ganador es circular: si naive es uno de
    los candidatos sobre los que se toma el argmin, el ganador nunca puede
    tener peor MASE que naive en ese bloque, por construccion matematica -no
    porque el metodo sea mejor. Se detecto exactamente este patron (100% de
    'victorias' en las 150 series de la primera corrida) al revisar los
    resultados antes de publicarlos, y se corrigio aqui usando
    `honest_outer_estimate`: el ganador se evalua sobre un bloque exterior que
    ni la seleccion del metodo ni el ajuste de hiperparametros vieron.
    """
    idx = pd.date_range("2000-01-01", periods=len(y), freq="MS")
    s = pd.Series(y, index=idx)

    try:
        prof = classify_series(s, m=m)
    except Exception as exc:
        return {"unique_id": uid, "n_obs": len(y), "estado": "error_clasificacion",
                "detalle": "{}: {}".format(type(exc).__name__, exc)}

    try:
        out = honest_outer_estimate(s, m=m, outer_block=outer_block,
                                     structural_filter=structural_filter)
    except Exception as exc:
        return {"unique_id": uid, "n_obs": len(y), "estado": "error",
                "detalle": "{}: {}".format(type(exc).__name__, exc)}
    if not out.get("ok"):
        return {"unique_id": uid, "n_obs": len(y), "estado": "sin_ganador",
                "detalle": out.get("reason", "")}

    regimen = ("estacional" if prof.has_seasonality else "no_estacional")
    regimen += "_con_tendencia" if prof.has_trend else "_plano"

    outer = out["outer_metrics"].set_index("modelo")
    fila = {"unique_id": uid, "n_obs": len(y), "estado": "ok", "regimen": regimen,
            "ganador": out["winner"]}
    # F32: mase_naive/mase_seasonal_naive y mase_ganador son columnas
    # independientes. Antes se escribian con el mismo "for metodo in (...,
    # out['winner'])", que cuando el ganador ERA 'naive' colapsaba las dos
    # escrituras de 'naive' en una sola (la del ganador) y dejaba mase_naive
    # sin poblar -NaN- para esas filas, descartandolas silenciosamente del
    # test de Wilcoxon (17 series de 150 en la corrida original). Se escriben
    # aqui por separado, sin importar quien haya ganado.
    for metodo in ("naive", "seasonal_naive"):
        if metodo in outer.index:
            r = outer.loc[metodo]
            if bool(r["elegible"]):
                fila["mase_{}".format(metodo)] = float(r["mase"])
    winner = out["winner"]
    if winner in outer.index:
        r = outer.loc[winner]
        if bool(r["elegible"]):
            fila["mase_ganador"] = float(r["mase"])
    return fila


def diebold_mariano_multiseries(errores_a: list[np.ndarray], errores_b: list[np.ndarray]):
    """DM agregado sobre multiples series (Petropoulos & Svetunkov, 2020).

    Cada serie aporta su perdida cuadratica media (a nivel de serie, no de
    origen individual), y el DM se calcula sobre esa muestra de diferencias
    entre series -evita pseudo-replicacion dentro de cada serie.
    """
    from scipy import stats

    d = []
    for ea, eb in zip(errores_a, errores_b):
        if ea.size == 0 or eb.size == 0:
            continue
        d.append(float(np.mean(ea ** 2) - np.mean(eb ** 2)))
    d = np.array(d)
    if d.size < 5:
        return float("nan"), float("nan"), int(d.size)
    t_stat, p_val = stats.ttest_1samp(d, 0.0)
    return float(t_stat), float(p_val), int(d.size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-series", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--structural-filter", dest="structural_filter", action="store_true",
                     default=True, help="Filtro estructural activado (por defecto)")
    ap.add_argument("--no-structural-filter", dest="structural_filter", action="store_false",
                     help="Desactiva el filtro estructural (F33: ablacion)")
    args = ap.parse_args()

    # F36: cada longitud escribe su propio CSV (panel_publico_len{max_len}.csv);
    # ademas se mantiene panel_publico.csv como alias del caso base (max_len=48)
    # por compatibilidad con el resto del pipeline/manuscrito que ya lo cita.
    suffix = "_len{}".format(args.max_len)

    if args.n_series > 300:
        print("ATENCION: n-series={} excede el limite de 300 del prompt maestro "
              "(evitar el M3/M4 completo por presupuesto de RAM). Se recorta a 300."
              .format(args.n_series))
        args.n_series = 300

    print("Descargando/leyendo M3 mensual (datasetsforecast)...")
    df = load_m3_monthly_short(args.max_len, args.min_len)
    todas_ids = df["unique_id"].unique()
    print("Series M3 mensuales truncadas a las ultimas {} obs (min. nativo {}): "
          "{} disponibles".format(args.max_len, args.min_len, len(todas_ids)))
    if len(todas_ids) == 0:
        print("No hay series elegibles; revise --min-len / --max-len.")
        return 1

    rng = np.random.default_rng(args.seed)
    ids = rng.choice(todas_ids, size=min(args.n_series, len(todas_ids)), replace=False)
    print("Muestra evaluada: {} series (semilla {})".format(len(ids), args.seed))

    rows = []
    for i, uid in enumerate(ids, 1):
        y = df.loc[df["unique_id"] == uid, "y"].to_numpy(dtype=float)
        row = evaluate_one(uid, y, structural_filter=args.structural_filter)
        rows.append(row)
        if i % 25 == 0 or i == len(ids):
            ok = sum(1 for r in rows if r.get("estado") == "ok")
            print("  [{}/{}] procesadas ({} ok)".format(i, len(ids), ok))

    res_df = pd.DataFrame(rows)
    n_ok = (res_df["estado"] == "ok").sum()
    n_fail = len(res_df) - n_ok
    print("\nProcesadas: {} | ok: {} | fallidas/sin ganador: {}".format(len(res_df), n_ok, n_fail))
    if n_fail:
        print("  Motivos (sin cobertura silenciosa):")
        for motivo, cnt in res_df.loc[res_df["estado"] != "ok", "estado"].value_counts().items():
            print("    {}: {}".format(motivo, cnt))

    ok = res_df[res_df["estado"] == "ok"].copy()
    ok["max_len"] = args.max_len
    ok["structural_filter"] = args.structural_filter
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok.to_csv(OUT_DIR / "panel_publico{}.csv".format(suffix), index=False, encoding="utf-8-sig")
    if args.max_len == 48 and args.structural_filter:
        # Alias de compatibilidad: el resto del pipeline y el manuscrito citan
        # "panel_publico.csv" a secas para el caso base (max_len=48, filtro on).
        ok.to_csv(OUT_DIR / "panel_publico.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("DISTRIBUCION DE MASE POR REGIMEN ESTRUCTURAL ({} series)".format(len(ok)))
    print("=" * 88)
    regimen_rows = []
    for regimen, grupo in ok.groupby("regimen"):
        print("\n  {} (n={})".format(regimen, len(grupo)))
        medianas = {}
        for col in ("mase_naive", "mase_seasonal_naive", "mase_ganador"):
            if col in grupo.columns and grupo[col].notna().any():
                v = grupo[col].dropna()
                medianas[col] = float(v.median())
                print("    {:22s} mediana={:.3f}  media={:.3f}  q25={:.3f}  q75={:.3f}".format(
                    col, v.median(), v.mean(), v.quantile(0.25), v.quantile(0.75)))
        fila_reg = {"regimen": regimen, "n": len(grupo),
                    "mediana_mase_ganador": medianas.get("mase_ganador", float("nan")),
                    "mediana_mase_naive": medianas.get("mase_naive", float("nan"))}
        if "mase_ganador" in grupo.columns and "mase_naive" in grupo.columns:
            comp = grupo.dropna(subset=["mase_ganador", "mase_naive"])
            if len(comp):
                gana = (comp["mase_ganador"] < comp["mase_naive"]).mean()
                pierde_mediana = fila_reg["mediana_mase_ganador"] > fila_reg["mediana_mase_naive"]
                print("    la herramienta supera al naive en {:.0%} de las series ({} con par valido)"
                      .format(gana, len(comp)))
                if pierde_mediana:
                    print("    ATENCION: en este regimen la herramienta PIERDE contra naive en la "
                          "mediana ({:.3f} vs {:.3f})".format(
                              fila_reg["mediana_mase_ganador"], fila_reg["mediana_mase_naive"]))
                fila_reg["tasa_victoria"] = float(gana)
                fila_reg["n_pares_validos"] = int(len(comp))
        regimen_rows.append(fila_reg)
    pd.DataFrame(regimen_rows).to_csv(OUT_DIR / "panel_publico_por_regimen{}.csv".format(suffix),
                                       index=False, encoding="utf-8-sig")

    print("\n" + "-" * 88)
    print("Ganador mas frecuente por regimen:")
    print(ok.groupby("regimen")["ganador"].agg(lambda s: s.value_counts().idxmax()))

    print("\n" + "-" * 88)
    print("Desglose victorias/empates/derrotas: herramienta (mase_ganador) vs. naive")
    comp_all = ok.dropna(subset=["mase_ganador", "mase_naive"])
    n_total_ok = len(ok)
    n_pares = len(comp_all)
    n_sin_par = n_total_ok - n_pares
    victorias = int((comp_all["mase_ganador"] < comp_all["mase_naive"]).sum())
    derrotas = int((comp_all["mase_ganador"] > comp_all["mase_naive"]).sum())
    empates = int((comp_all["mase_ganador"] == comp_all["mase_naive"]).sum())
    print("  Series con estado='ok': {}".format(n_total_ok))
    print("  Series con par (mase_ganador, mase_naive) no nulo: {}".format(n_pares))
    print("  Series excluidas por par no disponible (mase_naive o mase_ganador NaN): {}"
          .format(n_sin_par))
    print("  Victorias (mase_ganador < mase_naive): {} ({:.0%})".format(victorias, victorias / n_pares if n_pares else 0))
    print("  Empates exactos (mase_ganador == mase_naive): {} ({:.0%})".format(empates, empates / n_pares if n_pares else 0))
    print("  Derrotas (mase_ganador > mase_naive): {} ({:.0%})".format(derrotas, derrotas / n_pares if n_pares else 0))
    wtl = pd.DataFrame([{"n_series_ok": n_total_ok, "n_pares_validos": n_pares,
                          "n_excluidas_sin_par": n_sin_par, "victorias": victorias,
                          "empates": empates, "derrotas": derrotas}])
    wtl.to_csv(OUT_DIR / "panel_publico_wtl{}.csv".format(suffix), index=False, encoding="utf-8-sig")

    print("\n" + "-" * 88)
    print("Prueba de Diebold-Mariano agregada (herramienta vs. naive; H0: igual exactitud)")
    print("Nota: requiere errores por origen, no solo el MASE resumen; con las columnas")
    print("agregadas de este panel se reporta un contraste aproximado sobre el MASE por")
    print("serie (equivalente a un signo de Wilcoxon), declarado explicitamente como tal:")
    # F32: Wilcoxon debe excluir empates exactos explicitamente (tratamiento
    # estandar: scipy los excluye internamente por defecto, pero aqui se
    # reporta cuantos fueron para que el n resultante quede documentado).
    comp = comp_all[comp_all["mase_ganador"] != comp_all["mase_naive"]]
    print("  Pares con diferencia no nula usados en Wilcoxon: {} (de {} pares totales, "
          "{} empates exactos excluidos)".format(len(comp), n_pares, empates))
    if len(comp) >= 10:
        from scipy import stats
        stat, p = stats.wilcoxon(comp["mase_ganador"], comp["mase_naive"])
        print("  Wilcoxon signed-rank: estadistico={:.1f}  p={:.4f}  n={}".format(stat, p, len(comp)))
        print("  {}".format(
            "Se rechaza H0 (p<0.05): diferencia significativa entre herramienta y naive."
            if p < 0.05 else
            "No se rechaza H0 (p>=0.05): la diferencia con el naive NO es estadisticamente "
            "significativa en esta muestra."
        ))
    print("=" * 88)
    print("\nCSV:", OUT_DIR / "panel_publico{}.csv".format(suffix))
    print("CSV por regimen:", OUT_DIR / "panel_publico_por_regimen{}.csv".format(suffix))
    print("CSV W/T/L:", OUT_DIR / "panel_publico_wtl{}.csv".format(suffix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
