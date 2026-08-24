"""F06 y F12 - metricas correctas, MAPE que no explota, ME implementado."""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.metrics import (
    compute_metrics, mad, mape, mase, me, mse, rmse, seasonal_naive_scale,
    smape, tracking_signal,
)


# ---------------------------------------------------------------------------
# Valores calculados a mano
# ---------------------------------------------------------------------------
Y_TRUE = [100.0, 200.0, 300.0, 400.0]
Y_PRED = [110.0, 190.0, 330.0, 360.0]
# errores: -10, +10, -30, +40


def test_mad_valor_conocido():
    assert mad(Y_TRUE, Y_PRED) == pytest.approx((10 + 10 + 30 + 40) / 4)


def test_mse_y_rmse_valor_conocido():
    esperado = (100 + 100 + 900 + 1600) / 4
    assert mse(Y_TRUE, Y_PRED) == pytest.approx(esperado)
    assert rmse(Y_TRUE, Y_PRED) == pytest.approx(np.sqrt(esperado))


def test_me_valor_conocido_y_signo():
    """F06: el ME existe y su signo indica la direccion del sesgo."""
    assert me(Y_TRUE, Y_PRED) == pytest.approx((-10 + 10 - 30 + 40) / 4)
    # Sobreestimar sistematicamente => ME negativo => riesgo de sobrestock.
    assert me([100.0, 100.0], [120.0, 130.0]) < 0
    # Subestimar sistematicamente => ME positivo => riesgo de faltante.
    assert me([100.0, 100.0], [80.0, 70.0]) > 0


def test_mape_valor_conocido():
    esperado = 100 * np.mean([10 / 100, 10 / 200, 30 / 300, 40 / 400])
    assert mape(Y_TRUE, Y_PRED) == pytest.approx(esperado)


def test_mad_usa_valor_absoluto_no_es_el_error_medio():
    """F23: la tesis imprime MAD sin barras de valor absoluto (seria el ME)."""
    yt, yp = [100.0, 100.0], [80.0, 120.0]
    assert mad(yt, yp) == pytest.approx(20.0)
    assert me(yt, yp) == pytest.approx(0.0)
    assert mad(yt, yp) != me(yt, yp)


# ---------------------------------------------------------------------------
# F12 - MAPE ante demanda cero
# ---------------------------------------------------------------------------
def test_mape_no_explota_con_demanda_cero():
    """El codigo original devolvia 12 500 000 003 % para este caso."""
    yt = [100.0, 120.0, 0.0, 110.0]
    yp = [105.0, 115.0, 5.0, 108.0]
    valor, excluidos = mape(yt, yp, return_excluded=True)
    assert excluidos == 1
    assert valor < 100, "MAPE = {} : el cero sigue dominando la metrica".format(valor)
    assert len(str(int(valor))) <= 4


def test_mape_es_nan_si_toda_la_demanda_es_cero():
    valor, excluidos = mape([0.0, 0.0], [1.0, 2.0], return_excluded=True)
    assert np.isnan(valor) and excluidos == 2


def test_smape_definido_con_ceros():
    assert np.isfinite(smape([0.0, 100.0], [0.0, 90.0]))


# ---------------------------------------------------------------------------
# MASE
# ---------------------------------------------------------------------------
def test_mase_del_naive_es_uno_en_su_propio_train():
    """Un naive sobre datos con la misma escala da MASE ~ 1: es la referencia."""
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0])
    escala = seasonal_naive_scale(y, m=1)
    assert escala == pytest.approx(np.mean(np.abs(np.diff(y))))
    valor = mase(y[1:], y[:-1], y, m=1)
    assert valor == pytest.approx(1.0, rel=1e-9)


def test_mase_menor_que_uno_si_el_modelo_supera_al_naive():
    y_train = np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
    y_true = np.array([15.0, 15.0])
    assert mase(y_true, [15.0, 15.0], y_train, m=1) < 1.0


def test_mase_es_nan_si_la_serie_de_referencia_es_constante():
    assert np.isnan(mase([1.0, 2.0], [1.0, 2.0], np.ones(10), m=1))


def test_mase_es_escala_independiente():
    """La misma serie multiplicada por 1000 debe dar el mismo MASE."""
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0])
    a = mase(y[1:], y[:-1] * 1.05, y, m=1)
    b = mase(y[1:] * 1000, y[:-1] * 1.05 * 1000, y * 1000, m=1)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Senal de rastreo y conjunto completo
# ---------------------------------------------------------------------------
def test_senal_de_rastreo_detecta_sesgo_sistematico():
    yt = np.full(12, 100.0)
    sesgado = np.full(12, 80.0)     # subestima siempre
    insesgado = np.array([90.0, 110.0] * 6)
    assert abs(tracking_signal(yt, sesgado)) > 4
    assert abs(tracking_signal(yt, insesgado)) < 4


def test_compute_metrics_devuelve_el_conjunto_completo():
    ms = compute_metrics(Y_TRUE, Y_PRED, np.arange(1.0, 20.0), m=1)
    for campo in ("mase", "mape", "mad", "mse", "rmse", "me", "smape",
                  "tracking_signal", "n_preds", "mape_n_excluded"):
        assert hasattr(ms, campo)
    assert ms.n_preds == 4


def test_las_predicciones_no_finitas_se_descartan_no_propagan_nan():
    valor = mad([1.0, 2.0, 3.0], [1.0, np.nan, 3.0])
    assert np.isfinite(valor) and valor == pytest.approx(0.0)
