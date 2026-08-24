"""F19 - lote multi-SKU con memoria acotada, y F20/F21 - intervalos e inventario.

Presupuesto objetivo (Ryzen 5 7000, 8 GB): la memoria residente del lote no
puede crecer de forma monotona con el numero de SKU.
"""
from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from forecasting_core.batch import BatchConfig, resolve_n_jobs, run_batch
from forecasting_core.intervals import prediction_interval
from forecasting_core.inventory import compute_policy, safety_stock
from forecasting_core.models import get_spec
from forecasting_core.optimize import run_pipeline
from conftest import make_series


MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _panel(n_skus: int, n_obs: int = 36) -> pd.DataFrame:
    tipos = ["plana", "tendencia", "estacional", "tendencia_estacional", "paseo_aleatorio"]
    marcos = []
    for i in range(n_skus):
        s = make_series(tipos[i % len(tipos)], n_obs, seed=1000 + i)
        marcos.append(pd.DataFrame({
            "sku": "SKU-{:03d}".format(i),
            "year": s.index.year,
            "month": [MESES[m - 1] for m in s.index.month],
            "demand": np.round(np.abs(s.to_numpy()), 2),
        }))
    return pd.concat(marcos, ignore_index=True)


# ---------------------------------------------------------------------------
# Paralelismo acotado por RAM
# ---------------------------------------------------------------------------
def test_n_jobs_nunca_usa_todos_los_nucleos():
    n = resolve_n_jobs()
    assert 1 <= n <= 4, "n_jobs={} : en un equipo de 8 GB esto agota la memoria".format(n)


def test_n_jobs_se_acota_por_memoria_disponible():
    assert resolve_n_jobs(available_mb=500) == 1
    assert resolve_n_jobs(available_mb=100) == 1
    assert resolve_n_jobs(requested=1) == 1


# ---------------------------------------------------------------------------
# Memoria del lote
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_la_memoria_del_lote_no_crece_con_el_numero_de_skus(tmp_path):
    cfg = BatchConfig(horizon=6, flush_every=5, with_inventory=False)

    picos = {}
    for n_skus in (6, 24):
        gc.collect()
        tracemalloc.start()
        run_batch(_panel(n_skus, 36), tmp_path / str(n_skus), cfg)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        picos[n_skus] = pico / 1e6
        gc.collect()

    crecimiento = picos[24] / max(picos[6], 1e-9)
    assert crecimiento < 2.5, (
        "la memoria crecio {:.1f}x al cuadruplicar los SKU ({:.1f} MB -> {:.1f} MB): "
        "el lote esta acumulando en memoria en vez de volcar a disco".format(
            crecimiento, picos[6], picos[24])
    )


def test_el_lote_vuelca_resultados_a_disco(tmp_path):
    resumen = run_batch(_panel(4, 36), tmp_path, BatchConfig(horizon=6, flush_every=2))
    assert resumen.output_path.exists()
    df = pd.read_csv(resumen.output_path)
    assert len(df) == 4
    assert {"sku", "estado", "metodo", "mase", "me"} <= set(df.columns)
    assert resumen.peak_rows_in_memory <= 2 + 1


def test_el_lote_reporta_si_el_ganador_supera_al_naive(tmp_path):
    """Sin esa columna, un MASE aislado no dice nada (F12)."""
    resumen = run_batch(_panel(4, 48), tmp_path, BatchConfig(horizon=6))
    df = pd.read_csv(resumen.output_path)
    assert "supera_naive" in df.columns
    assert "mase_naive" in df.columns


def test_un_sku_defectuoso_no_tumba_el_lote(tmp_path):
    panel = _panel(3, 36)
    malo = pd.DataFrame({"sku": "SKU-MALO", "year": [2023] * 4,
                         "month": ["enero", "febrero", "marzo", "abril"],
                         "demand": [1.0, 2.0, 3.0, 4.0]})
    resumen = run_batch(pd.concat([panel, malo], ignore_index=True), tmp_path,
                        BatchConfig(horizon=6))
    assert resumen.n_skus == 4
    assert resumen.n_ok == 3
    assert "SKU-MALO" in resumen.failures


def test_el_lote_exporta_pronosticos_con_intervalos(tmp_path):
    resumen = run_batch(_panel(3, 48), tmp_path, BatchConfig(horizon=6, level=0.95))
    assert resumen.forecast_path.exists()
    df = pd.read_csv(resumen.forecast_path)
    assert {"sku", "fecha", "pronostico", "inferior_95", "superior_95"} <= set(df.columns)
    assert (df["superior_95"] >= df["pronostico"]).all()
    assert (df["pronostico"] >= df["inferior_95"]).all()


# ---------------------------------------------------------------------------
# F20 - intervalos
# ---------------------------------------------------------------------------
def test_el_intervalo_contiene_al_pronostico_y_respeta_el_piso_en_cero():
    s = make_series("tendencia_estacional", 60)
    res = run_pipeline(s, m=12)
    pi = prediction_interval(s, get_spec(res.winner), res.winner_params,
                             season_length=12, horizon=12, level=0.95)
    assert (pi.lower <= pi.mean + 1e-9).all()
    assert (pi.upper >= pi.mean - 1e-9).all()
    assert (pi.lower >= 0).all(), "el limite inferior de una demanda no puede ser negativo"
    assert len(pi.index) == 12
    assert pi.method


def test_la_banda_se_ensancha_cuando_el_error_se_acumula():
    """En un paseo aleatorio la varianza del error crece con h: la banda debe abrirse.

    No es una propiedad universal: una regresion lineal bien especificada sobre
    una tendencia deterministica tiene error aproximadamente constante en h, y
    ahi la banda NO se ensancha. Los cuantiles empiricos capturan la diferencia
    en vez de imponer un sqrt(h) que seria falso en el segundo caso.
    """
    s = make_series("paseo_aleatorio", 72)
    pi = prediction_interval(s, get_spec("naive"), {}, season_length=12,
                             horizon=12, level=0.95)
    ancho = pi.upper - pi.lower
    assert ancho[-1] > ancho[0], "la incertidumbre debe crecer con el horizonte"
    assert pi.sigma[-1] > pi.sigma[0]


def test_regresion_sobre_tendencia_no_infla_la_banda_artificialmente():
    """El contraejemplo: aqui un sqrt(h) impuesto exageraria la incertidumbre."""
    s = make_series("tendencia_limpia", 72)
    pi = prediction_interval(s, get_spec("linear_regression"), {}, season_length=12,
                             horizon=12, level=0.95)
    ancho = pi.upper - pi.lower
    assert ancho.max() / max(ancho.min(), 1e-9) < 5.0


def test_un_nivel_mayor_da_una_banda_mas_ancha():
    s = make_series("plana", 60)
    spec = get_spec("ses")
    a = prediction_interval(s, spec, {"alpha": 0.3}, season_length=12, horizon=6, level=0.80)
    b = prediction_interval(s, spec, {"alpha": 0.3}, season_length=12, horizon=6, level=0.99)
    assert ((b.upper - b.lower) >= (a.upper - a.lower) - 1e-9).all()


# ---------------------------------------------------------------------------
# F20 / M-08 - inventario
# ---------------------------------------------------------------------------
def test_stock_de_seguridad_crece_con_el_nivel_de_servicio():
    ss90, z90 = safety_stock(100.0, 0.90)
    ss99, z99 = safety_stock(100.0, 0.99)
    assert ss99 > ss90 > 0 and z99 > z90
    assert z90 == pytest.approx(1.2816, abs=1e-3)


def test_nivel_de_servicio_invalido_es_error():
    with pytest.raises(ValueError):
        safety_stock(100.0, 1.5)


def test_la_politica_produce_punto_de_reorden_coherente():
    s = make_series("tendencia_estacional", 72)
    res = run_pipeline(s, m=12)
    pol = compute_policy(s, get_spec(res.winner), res.winner_params,
                         lead_time=3, service_level=0.95, season_length=12)
    assert pol.demand_lead_time > 0
    assert pol.sigma_lead_time > 0
    assert pol.safety_stock > 0
    assert pol.reorder_point == pytest.approx(pol.demand_lead_time + pol.safety_stock)
    assert pol.sigma_method, "el metodo de estimacion de sigma debe quedar declarado"


def test_la_politica_avisa_si_el_modelo_esta_sesgado():
    """El stock de seguridad protege contra variabilidad, no contra sesgo."""
    idx = pd.date_range("2019-01-01", periods=60, freq="MS")
    s = pd.Series(np.linspace(1000, 3000, 60), index=idx)   # tendencia fuerte
    pol = compute_policy(s, get_spec("mean"), {}, lead_time=6,
                         service_level=0.95, season_length=12)
    assert abs(pol.bias_lead_time) > 0
    assert any("sesgo" in w for w in pol.warnings)
