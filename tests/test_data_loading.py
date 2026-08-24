"""F15 - carga robusta: las siete variantes de entrada, con comportamiento declarado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_core.data import (
    MIN_OBS_REQUIRED, load_panel, load_series, normalize_month,
)


def _panel(months, demands=None, years=None, sku=None):
    n = len(months)
    demands = demands if demands is not None else list(range(100, 100 + n))
    years = years if years is not None else [2023 + i // 12 for i in range(n)]
    data = {"year": years, "month": months, "demand": demands}
    if sku is not None:
        data["sku"] = sku
    return pd.DataFrame(data)


def _meses_es(n):
    base = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
            "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return [base[i % 12] for i in range(n)]


# ---------------------------------------------------------------------------
# 1. meses numericos  2. meses en ingles  3. abreviaturas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("valor,esperado", [
    (1, 1), (12, 12), ("3", 3), (7.0, 7), ("07", 7),
    ("enero", 1), ("Enero", 1), ("ENERO", 1), ("diciembre", 12),
    ("setiembre", 9), ("septiembre", 9),
    ("January", 1), ("december", 12), ("May", 5),
    ("ene", 1), ("dic", 12), ("jan", 1), ("dec", 12), ("sept", 9),
    ("marzo.", 3), ("  abril  ", 4),
    (0, None), (13, None), ("basura", None), ("", None), (None, None),
])
def test_normalizacion_de_meses(valor, esperado):
    assert normalize_month(valor) == esperado


def test_carga_con_meses_numericos():
    """El codigo original solo aceptaba nombres en espanol: un ERP produce 1-12."""
    df = _panel(list(range(1, 13)) * 2)
    res = load_series(df)
    assert res.report.ok, res.report.errors
    assert res.report.n_obs == 24


def test_carga_con_meses_en_ingles():
    ingles = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    res = load_series(_panel(ingles * 2))
    assert res.report.ok, res.report.errors
    assert res.report.n_obs == 24


def test_mes_no_reconocido_produce_error_explicito():
    meses = _meses_es(24)
    meses[5] = "Mes6"
    res = load_series(_panel(meses))
    assert not res.report.ok
    assert "mes no reconocido" in " ".join(res.report.errors)
    assert "Mes6" in " ".join(res.report.unparsed_months)


# ---------------------------------------------------------------------------
# 4. separador de miles / decimal
# ---------------------------------------------------------------------------
def test_separador_decimal_coma():
    res = load_series(_panel(_meses_es(24), demands=["1234,56"] * 24))
    assert res.report.ok, res.report.errors
    assert res.series.dropna().iloc[0] == pytest.approx(1234.56)
    assert any("coma = separador decimal" in n for n in res.report.notes)


def test_separador_de_miles_europeo():
    res = load_series(_panel(_meses_es(24), demands=["1.234,56"] * 24))
    assert res.report.ok, res.report.errors
    assert res.series.dropna().iloc[0] == pytest.approx(1234.56)
    assert any("europeo" in n for n in res.report.notes)


def test_separador_de_miles_anglosajon():
    res = load_series(_panel(_meses_es(24), demands=["1,234.56"] * 24))
    assert res.report.ok, res.report.errors
    assert res.series.dropna().iloc[0] == pytest.approx(1234.56)
    assert any("anglosajon" in n for n in res.report.notes)


def test_la_convencion_asumida_siempre_queda_declarada():
    """Nunca adivinar en silencio."""
    res = load_series(_panel(_meses_es(24), demands=["1.234,56"] * 24))
    assert res.report.notes, "la conversion numerica no dejo rastro en el reporte"


def test_demanda_no_numerica_es_error():
    d = [100.0] * 24
    d[3] = "N/D"
    res = load_series(_panel(_meses_es(24), demands=d))
    assert not res.report.ok
    assert "no numerica" in " ".join(res.report.errors)


# ---------------------------------------------------------------------------
# 5. duplicados
# ---------------------------------------------------------------------------
def test_duplicados_identicos_se_consolidan_y_se_informa():
    meses = _meses_es(24) + ["enero"]
    years = [2023 + i // 12 for i in range(24)] + [2023]
    demandas = list(range(100, 124)) + [100]
    res = load_series(_panel(meses, demands=demandas, years=years))
    assert res.report.ok, res.report.errors
    assert res.report.duplicates
    assert any("duplicados" in n for n in res.report.notes)


def test_duplicados_contradictorios_son_error_no_se_elige_por_el_usuario():
    meses = _meses_es(24) + ["enero"]
    years = [2023 + i // 12 for i in range(24)] + [2023]
    demandas = list(range(100, 124)) + [999]
    res = load_series(_panel(meses, demands=demandas, years=years))
    assert not res.report.ok
    assert "DISTINTOS" in " ".join(res.report.errors)


# ---------------------------------------------------------------------------
# 6. huecos temporales (el defecto central de F15)
# ---------------------------------------------------------------------------
def _con_hueco():
    meses = _meses_es(30)
    years = [2023 + i // 12 for i in range(30)]
    df = _panel(meses, years=years)
    return df.drop(index=[10, 11]).reset_index(drop=True)   # faltan 2 meses


def test_huecos_se_reportan_y_no_se_colapsan_en_silencio():
    """El original hacia asfreq + dropna: dos meses no adyacentes pasaban a serlo."""
    res = load_series(_con_hueco(), gap_policy="report")
    assert res.report.gaps, "los huecos no fueron detectados"
    assert any("faltantes" in n for n in res.report.notes)
    # El indice conserva la rejilla mensual completa: no se desalinea el lag 12.
    assert res.series.index.freq == "MS"
    assert res.series.isna().sum() == len(res.report.gaps)


def test_politica_interpolar():
    res = load_series(_con_hueco(), gap_policy="interpolate")
    assert res.report.ok, res.report.errors
    assert res.series.isna().sum() == 0
    assert any("interpolados" in n for n in res.report.notes)


def test_politica_cero():
    res = load_series(_con_hueco(), gap_policy="zero")
    assert res.series.isna().sum() == 0
    assert (res.series == 0).sum() == 2


def test_politica_abortar():
    res = load_series(_con_hueco(), gap_policy="abort")
    assert not res.report.ok
    assert "faltantes" in " ".join(res.report.errors)


# ---------------------------------------------------------------------------
# 7. validaciones generales
# ---------------------------------------------------------------------------
def test_faltan_columnas_obligatorias():
    df = pd.DataFrame({"anio": [2023], "mes": ["enero"], "cantidad": [10]})
    res = load_series(df)
    assert not res.report.ok
    assert "year" in " ".join(res.report.errors)


def test_serie_demasiado_corta_es_error_declarado():
    res = load_series(_panel(_meses_es(6)))
    assert not res.report.ok
    assert str(MIN_OBS_REQUIRED) in " ".join(res.report.errors)


def test_ceros_y_negativos_se_informan():
    d = [100.0] * 24
    d[2] = 0.0
    d[5] = -30.0
    res = load_series(_panel(_meses_es(24), demands=d))
    assert res.report.zero_values == 1
    assert res.report.negative_values == 1
    assert any("cero" in n for n in res.report.notes)
    assert any("negativos" in n for n in res.report.notes)


def test_encabezados_con_mayusculas_y_espacios():
    df = _panel(_meses_es(24))
    df.columns = [" YEAR ", "Month", "DEMAND"]
    assert load_series(df).report.ok


def test_carga_de_panel_multi_sku():
    """F19: la operacion real tiene multiples referencias."""
    a = _panel(_meses_es(24), sku=["SKU-A"] * 24)
    b = _panel(_meses_es(24), demands=list(range(500, 524)), sku=["SKU-B"] * 24)
    panel = pd.concat([a, b], ignore_index=True)
    res = load_panel(panel)
    assert set(res) == {"SKU-A", "SKU-B"}
    assert all(r.report.ok for r in res.values())
    assert res["SKU-B"].series.dropna().iloc[0] == pytest.approx(500.0)
