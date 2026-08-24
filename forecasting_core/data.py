"""Carga y validacion de series de demanda.

Resuelve F15 (carga fragil: meses numericos, meses en ingles, separador de miles,
duplicados, huecos temporales colapsados en silencio).

Principio: nada se corrige en silencio. Toda normalizacion o descarte queda
registrado en LoadReport y es visible para quien llama.
"""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "LoadReport", "LoadResult", "load_series", "load_series_from_excel",
    "load_panel", "normalize_month", "MIN_OBS_REQUIRED",
]

# Minimo de observaciones para permitir cualquier analisis.
MIN_OBS_REQUIRED = 18

GapPolicy = Literal["report", "interpolate", "zero", "abort"]

_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
# Abreviaturas de 3 letras de ambos idiomas (ene/jan, dic/dec, ...).
_MONTH_LOOKUP: dict[str, int] = {}
for _src in (_MONTHS_ES, _MONTHS_EN):
    for _name, _num in _src.items():
        _MONTH_LOOKUP[_name] = _num
        _MONTH_LOOKUP[_name[:3]] = _num
_MONTH_LOOKUP.update({"sept": 9, "set": 9, "ago": 8, "aug": 8})


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_month(value) -> int | None:
    """Convierte un mes a 1-12 desde numero, nombre ES/EN o abreviatura.

    Devuelve None si no se reconoce (quien llama decide si es error).
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        iv = int(value)
        return iv if 1 <= iv <= 12 else None
    if isinstance(value, (float, np.floating)):
        iv = int(round(float(value)))
        return iv if 1 <= iv <= 12 else None
    text = _strip_accents(str(value).strip().lower())
    text = re.sub(r"[.\s]+$", "", text)
    if not text:
        return None
    if text.isdigit():
        iv = int(text)
        return iv if 1 <= iv <= 12 else None
    try:
        fv = float(text)
    except ValueError:
        pass
    else:
        iv = int(round(fv))
        return iv if 1 <= iv <= 12 else None
    return _MONTH_LOOKUP.get(text)


def _normalize_numeric_series(raw: pd.Series) -> tuple[pd.Series, list[str]]:
    """Normaliza demanda a float declarando la convencion asumida.

    Distingue separador de miles europeo ('1.234,56') del punto decimal
    ('1234.56') de forma EXPLICITA en lugar de adivinar en silencio.
    """
    notes: list[str] = []
    if pd.api.types.is_numeric_dtype(raw):
        return raw.astype(float), notes

    txt = raw.astype(str).str.strip()
    txt = txt.str.replace(r"[^\d,.\-+eE]", "", regex=True)

    has_comma = txt.str.contains(",", regex=False)
    has_dot = txt.str.contains(".", regex=False)
    both = has_comma & has_dot

    if both.any():
        sample = txt[both].iloc[0]
        if sample.rfind(",") > sample.rfind("."):
            notes.append(
                "Convencion asumida: punto = separador de miles, coma = decimal "
                "(formato europeo). Detectado en valores como '{}'.".format(sample)
            )
            txt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            notes.append(
                "Convencion asumida: coma = separador de miles, punto = decimal "
                "(formato anglosajon). Detectado en valores como '{}'.".format(sample)
            )
            txt = txt.str.replace(",", "", regex=False)
    elif has_comma.any():
        # Solo comas: decimal si hay <=2 digitos despues, si no separador de miles.
        frac = txt.str.extract(r",(\d+)$")[0].dropna()
        if len(frac) and (frac.str.len() <= 2).all():
            notes.append("Convencion asumida: coma = separador decimal.")
            txt = txt.str.replace(",", ".", regex=False)
        else:
            notes.append("Convencion asumida: coma = separador de miles.")
            txt = txt.str.replace(",", "", regex=False)

    return pd.to_numeric(txt, errors="coerce"), notes


@dataclass
class LoadReport:
    """Todo lo que ocurrio durante la carga. Nada queda oculto."""

    n_rows_input: int = 0
    n_obs: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    duplicates: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    gap_policy: str = "report"
    unparsed_months: list[str] = field(default_factory=list)
    non_numeric_demand: int = 0
    negative_values: int = 0
    zero_values: int = 0
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = ["{} observaciones".format(self.n_obs)]
        if self.start is not None:
            parts.append("{:%Y-%m} -> {:%Y-%m}".format(self.start, self.end))
        if self.gaps:
            parts.append("{} huecos temporales".format(len(self.gaps)))
        if self.duplicates:
            parts.append("{} duplicados".format(len(self.duplicates)))
        if self.zero_values:
            parts.append("{} ceros".format(self.zero_values))
        return " | ".join(parts)


@dataclass
class LoadResult:
    series: pd.Series
    report: LoadReport


def load_series(
    df: pd.DataFrame,
    *,
    gap_policy: GapPolicy = "report",
    sku: str | None = None,
) -> LoadResult:
    """Construye una serie mensual a partir de un DataFrame year/month/demand.

    Acepta el DataFrame directamente (testeable sin Excel). Columna `sku`
    opcional para filtrar un producto de un panel.
    """
    report = LoadReport(n_rows_input=len(df), gap_policy=gap_policy)
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]

    if sku is not None:
        if "sku" not in work.columns:
            report.errors.append(
                "Se pidio filtrar por sku pero el archivo no tiene columna 'sku'."
            )
            return LoadResult(pd.Series(dtype=float), report)
        work = work[work["sku"].astype(str).str.strip() == str(sku).strip()]

    required = {"year", "month", "demand"}
    missing = required - set(work.columns)
    if missing:
        report.errors.append(
            "Faltan columnas obligatorias: {}. Se requieren year, month, demand.".format(
                ", ".join(sorted(missing))
            )
        )
        return LoadResult(pd.Series(dtype=float), report)

    # --- mes -------------------------------------------------------------
    months = work["month"].map(normalize_month)
    bad_months = work.loc[months.isna(), "month"]
    if len(bad_months):
        report.unparsed_months = sorted({str(v) for v in bad_months.head(10)})
        report.errors.append(
            "{} filas con mes no reconocido (ej: {}). Se aceptan 1-12, nombres en "
            "espanol o ingles, y abreviaturas de 3 letras.".format(
                len(bad_months), ", ".join(report.unparsed_months[:3])
            )
        )
        return LoadResult(pd.Series(dtype=float), report)

    # --- anio ------------------------------------------------------------
    years = pd.to_numeric(work["year"], errors="coerce")
    if years.isna().any():
        report.errors.append(
            "{} filas con anio no numerico.".format(int(years.isna().sum()))
        )
        return LoadResult(pd.Series(dtype=float), report)

    # --- demanda ---------------------------------------------------------
    demand, notes = _normalize_numeric_series(work["demand"])
    report.notes.extend(notes)
    n_bad = int(demand.isna().sum())
    if n_bad:
        report.non_numeric_demand = n_bad
        report.errors.append("{} filas con demanda no numerica.".format(n_bad))
        return LoadResult(pd.Series(dtype=float), report)

    dates = pd.to_datetime(
        dict(year=years.astype(int), month=months.astype(int), day=1), errors="coerce"
    )
    if dates.isna().any():
        report.errors.append(
            "{} filas con fecha invalida.".format(int(dates.isna().sum()))
        )
        return LoadResult(pd.Series(dtype=float), report)

    tidy = pd.DataFrame(
        {"date": dates.to_numpy(), "demand": demand.to_numpy(dtype=float)}
    ).sort_values("date")

    # --- duplicados ------------------------------------------------------
    dup_mask = tidy["date"].duplicated(keep=False)
    if dup_mask.any():
        grouped = tidy[dup_mask].groupby("date")["demand"]
        report.duplicates = [
            "{:%Y-%m} ({} filas, valores {})".format(d, len(v), sorted(set(np.round(v, 3))))
            for d, v in grouped
        ]
        agreed = bool(grouped.nunique().eq(1).all())
        if agreed:
            report.notes.append(
                "{} meses duplicados con valores identicos: se conservo una fila "
                "por mes.".format(len(report.duplicates))
            )
            tidy = tidy.drop_duplicates(subset="date", keep="first")
        else:
            report.errors.append(
                "Meses duplicados con valores DISTINTOS: {}. Corrija el archivo; la "
                "herramienta no elige por usted.".format("; ".join(report.duplicates[:5]))
            )
            return LoadResult(pd.Series(dtype=float), report)

    series = pd.Series(
        tidy["demand"].to_numpy(dtype=float), index=pd.DatetimeIndex(tidy["date"])
    )
    series.index.name = "date"
    series.name = sku or "demand"

    # --- huecos temporales (F15: nunca colapsar en silencio) --------------
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    reindexed = series.reindex(full_idx)
    missing_idx = reindexed.index[reindexed.isna()]
    if len(missing_idx):
        report.gaps = ["{:%Y-%m}".format(d) for d in missing_idx]
        if gap_policy == "abort":
            report.errors.append(
                "La serie tiene {} meses faltantes ({}). Politica 'abort'.".format(
                    len(missing_idx), ", ".join(report.gaps[:6])
                )
            )
            return LoadResult(pd.Series(dtype=float), report)
        if gap_policy == "interpolate":
            reindexed = reindexed.interpolate(method="time", limit_direction="both")
            report.notes.append(
                "{} meses faltantes interpolados linealmente en el tiempo: {}".format(
                    len(missing_idx), ", ".join(report.gaps[:6])
                )
            )
        elif gap_policy == "zero":
            reindexed = reindexed.fillna(0.0)
            report.notes.append(
                "{} meses faltantes rellenados con 0 (demanda nula asumida): {}".format(
                    len(missing_idx), ", ".join(report.gaps[:6])
                )
            )
        else:  # "report"
            report.notes.append(
                "ATENCION: {} meses faltantes conservados como NaN ({}). Eliminarlos "
                "desalinearia el indice estacional; elija 'interpolate' o 'zero' para "
                "continuar.".format(len(missing_idx), ", ".join(report.gaps[:6]))
            )
    series = reindexed
    series.index.freq = "MS"
    series.index.name = "date"
    series.name = sku or "demand"

    finite = series.dropna()
    report.n_obs = int(len(finite))
    if report.n_obs:
        report.start, report.end = finite.index.min(), finite.index.max()
    report.negative_values = int((finite < 0).sum())
    report.zero_values = int((finite == 0).sum())
    if report.negative_values:
        report.notes.append(
            "{} valores negativos de demanda: revise el archivo "
            "(devoluciones?).".format(report.negative_values)
        )
    if report.zero_values:
        report.notes.append(
            "{} meses con demanda cero: se excluyen del calculo de MAPE y se usara "
            "MASE como metrica primaria.".format(report.zero_values)
        )
    if report.n_obs < MIN_OBS_REQUIRED:
        report.errors.append(
            "Solo {} observaciones utiles; se requieren al menos {} para una "
            "validacion honesta.".format(report.n_obs, MIN_OBS_REQUIRED)
        )

    return LoadResult(series, report)


def load_series_from_excel(
    source,
    *,
    gap_policy: GapPolicy = "report",
    sku: str | None = None,
    **read_kwargs,
) -> LoadResult:
    """Carga desde ruta, buffer o bytes de un .xlsx/.xls/.csv."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    name = str(getattr(source, "name", source)).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(source, dtype=str, **read_kwargs)
    else:
        df = pd.read_excel(source, dtype=str, **read_kwargs)
    return load_series(df, gap_policy=gap_policy, sku=sku)


def load_panel(
    df: pd.DataFrame, *, gap_policy: GapPolicy = "report"
) -> dict[str, LoadResult]:
    """Carga un panel multi-SKU. Devuelve {sku: LoadResult}, uno por referencia."""
    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    if "sku" not in work.columns:
        return {"__single__": load_series(work, gap_policy=gap_policy)}
    out: dict[str, LoadResult] = {}
    for sku in work["sku"].astype(str).str.strip().unique():
        out[sku] = load_series(work, gap_policy=gap_policy, sku=sku)
    return out
