# ======================================================
# dash.py — Pronósticos (modulos 0 + 2 + 3 + 4)
# ======================================================

import io
import base64
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from dash.exceptions import PreventUpdate
from pandas.tseries.offsets import MonthBegin

# -------------------------------------------------
# Configuración base
# -------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Motor de Pronósticos – Tuboplex"

# -------------------------------------------------
# Diccionario de meses (coincide con tu motor)
# -------------------------------------------------
MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12
}

MESES_ES = {1:"enero", 2:"febrero", 3:"marzo", 4:"abril", 5:"mayo", 6:"junio",
            7:"julio", 8:"agosto", 9:"septiembre", 10:"octubre", 11:"noviembre", 12:"diciembre"}


# -------------------------------------------------
# Carga de serie
# -------------------------------------------------
def cargar_serie(file_content) -> pd.Series:
    """Carga Excel con columnas year, month, demand."""
    content_type, content_string = file_content.split(',')
    decoded = base64.b64decode(content_string)

    df = pd.read_excel(io.BytesIO(decoded), converters={"demand": lambda x: str(x).replace(",", ".")})
    df.columns = [c.strip().lower() for c in df.columns]

    if not {"year", "month", "demand"}.issubset(df.columns):
        raise ValueError("El archivo debe tener columnas: year, month, demand")

    df["month"] = df["month"].astype(str).str.strip().str.lower().map(MESES)
    df["demand"] = df["demand"].astype(float)

    fechas = pd.to_datetime(dict(
        year=df["year"].astype(int),
        month=df["month"].astype(int),
        day=1
    ))
    serie = pd.Series(df["demand"].values, index=fechas).sort_index().asfreq("MS")
    return serie

# -------------------------------------------------
# Layout (Mód 0 → Mód 2 → Mód 3)
# -------------------------------------------------
app.layout = dbc.Container([
    html.H2("1️⃣ Carga y validación de datos", className="mt-3 mb-4"),

    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "📤 Arrastra o selecciona tu archivo Excel con columnas ",
            html.Code("year, month, demand")
        ]),
        style={
            'width': '100%', 'height': '90px', 'lineHeight': '90px',
            'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),

    html.Div(id="alerta", className="mt-4"),
    html.Div(id="preview", className="mt-3"),

    html.Hr(),
    html.H4("2️⃣ Clasificación y evaluación de métodos", className="mt-2"),
    dcc.Loading(html.Div(id="mod2-output", className="mt-3"), type="default"),

    html.Hr(),
    html.H4("3️⃣ Optimización de parámetros del método ganador", className="mt-2"),
    dcc.Loading(html.Div(id="mod3-output", className="mt-3"), type="default"),

    html.Hr(),
    html.H4("4️⃣ Forecast con el método ganador", className="mt-2"),
    html.P(
        "La gráfica muestra: demanda histórica (línea azul), pronóstico simulado sobre la historia (línea roja punteada) "
        " y pronóstico futuro (línea verde) con el método ganador.",
        className="text-muted"
    ),

dbc.Row([
    dbc.Col([
        html.Label("Horizonte del forecast:"),
        dcc.RadioItems(
            id="horizon-select",
            options=[
                {"label": "6 meses", "value": 6},
                {"label": "12 meses", "value": 12},
                {"label": "18 meses", "value": 18},
                {"label": "24 meses", "value": 24},
            ],
            value=12,
            inline=True
        ),
    ], width="auto"),
], className="mb-3"),

dcc.Loading(html.Div(id="forecast-plot", className="mt-2"), type="default"),
dcc.Loading(html.Div(id="forecast-table", className="mt-3"), type="default"),

# Botón de descarga + componente Download
dbc.Button("⬇️ Descargar forecast (Excel)", id="btn-download-forecast",
           color="primary", className="mt-3"),
dcc.Download(id="download-forecast"),


    # Store para pasar método/params del Módulo 3 → Módulo 4
    dcc.Store(id="best-model-store"),
])

# -------------------------------------------------
# Callback Módulo 0 — Validación y preview
# -------------------------------------------------
@app.callback(
    [Output("alerta", "children"),
     Output("preview", "children")],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def validar_y_mostrar(contents, filename):
    if contents is None:
        return "", ""

    try:
        serie = cargar_serie(contents)
        n = len(serie.dropna())
        inicio = serie.index.min().strftime('%Y-%m')
        fin = serie.index.max().strftime('%Y-%m')

        # Tabla de vista previa (primeros 12)
        df_preview = serie.reset_index()
        df_preview.columns = ["Fecha", "Demanda"]
        tabla = dash_table.DataTable(
            data=df_preview.head(12).to_dict("records"),
            columns=[{"name": i, "id": i} for i in df_preview.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            page_size=12
        )

        if n < 24:
            alerta = dbc.Alert(
                f"⚠️ La serie tiene solo {n} meses de datos ({inicio} → {fin}). "
                "Se requieren al menos 24 meses para generar pronósticos confiables.",
                color="warning",
                dismissable=True
            )
            return alerta, tabla

        # ✅ Caso válido (≥24): alerta + tabla + gráfico
        alerta = dbc.Alert(
            f"✅ Serie válida ({n} meses) — Periodo: {inicio} → {fin}.",
            color="success",
            dismissable=True
        )

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=serie.index,
                    y=serie.values,
                    mode="lines+markers",
                    name="Demanda"
                )
            ]
        )
        fig.update_layout(
            title="Demanda mensual (serie cargada)",
            xaxis_title="Fecha",
            yaxis_title="Demanda",
            margin=dict(l=10, r=10, t=50, b=10),
            height=360
        )

        grafico = dcc.Graph(id="grafico-demanda", figure=fig, style={"height": "360px"})

        return alerta, html.Div([
            grafico,
            html.Br(),
            tabla
        ])

    except Exception as e:
        return dbc.Alert(f"❌ Error: {e}", color="danger"), ""

# -------------------------------------------------
# Clasificación 
# -------------------------------------------------
def clasificar_serie(serie: pd.Series, umbral_autocorr: float = 0.30):
    vals = serie.dropna().values
    x = np.arange(len(vals))
    slope, intercept, r, p_tend, _ = stats.linregress(x, vals)
    tendencia = (p_tend < 0.05)

    try:
        ac12 = float(pd.Series(serie).autocorr(lag=12))
    except Exception:
        ac12 = np.nan
    estacionalidad = (abs(ac12) > umbral_autocorr) if not np.isnan(ac12) else False

    p_adf = np.nan
    estacionaria = False
    try:
        p_adf = adfuller(serie.dropna(), autolag="AIC")[1]
        estacionaria = (p_adf < 0.05)
    except Exception:
        pass

    return {
        "tendencia": "Sí" if tendencia else "No",
        "p_tendencia": round(p_tend, 4),
        "estacionalidad": "Sí" if estacionalidad else "No",
        "autocorr_12": round(ac12, 3) if not np.isnan(ac12) else None,
        "estacionaria": "Sí" if estacionaria else "No",
        "p_adf": round(p_adf, 4) if not np.isnan(p_adf) else None,
    }

# -------------------------------------------------
# Métricas
# -------------------------------------------------
def mse(y_true, y_pred): return float(np.mean((np.array(y_true) - np.array(y_pred))**2))
def mad(y_true, y_pred): return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
def mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

# -------------------------------------------------
# Criterio de "serie muy lineal"
# -------------------------------------------------
def es_muy_lineal(serie: pd.Series, r2_umbral: float = 0.90, p_umbral: float = 0.01, ac12_umbral: float = 0.10) -> bool:
    """
    True si la serie es fuertemente lineal (aceptamos métodos constantes):
    - R^2 >= 0.90 y p_tendencia < 0.01
    - |autocorr_12| < 0.10
    """
    vals = serie.dropna().values
    if len(vals) < 6:
        return False

    x = np.arange(len(vals))
    slope, intercept, r, p_tend, _ = stats.linregress(x, vals)
    r2 = r**2

    try:
        ac12 = float(pd.Series(serie).autocorr(lag=12))
    except Exception:
        ac12 = np.nan
    ac12_abs = abs(ac12) if not np.isnan(ac12) else 0.0

    return (r2 >= r2_umbral) and (p_tend < p_umbral) and (ac12_abs < ac12_umbral)

# -------------------------------------------------
# métodos (como en tu motor)
# -------------------------------------------------
def f_promedio_simple(train): return np.mean(train)
def f_promedio_movil(train, k=3): return np.mean(train[-k:]) if len(train) >= k else np.mean(train)
def f_promedio_movil_ponderado(train, weights=[0.5, 0.3, 0.2]):
    k = len(weights)
    if len(train) < k: return np.mean(train)
    return np.dot(train[-k:], weights[::-1])
def f_ses(train, alpha=0.1):
    model = SimpleExpSmoothing(train, initialization_method="heuristic").fit(smoothing_level=alpha, optimized=False)
    return float(model.forecast(1)[0])
def f_regresion_lineal(train):
    x = np.arange(len(train))
    slope, intercept, *_ = stats.linregress(x, train)
    return float(intercept + slope * len(train))
def f_holt(train, alpha=0.1, beta=0.1):
    model = ExponentialSmoothing(train, trend="add", initialization_method="heuristic")
    fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    return float(fit.forecast(1)[0])
def f_holt_winters(train, alpha=0.1, beta=0.1, gamma=0.1, m=12):
    if len(train) < m + 2: return f_holt(train, alpha, beta)
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=m, initialization_method="heuristic")
    fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
    return float(fit.forecast(1)[0])
def f_arima(train):
    if len(train) < 6: return train[-1]
    mod = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    return float(res.forecast(1)[0])
def f_sarima(train, m=12):
    if len(train) < m + 2: return f_arima(train)
    mod = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,m),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    return float(res.forecast(1)[0])

METHODS = {
    "Promedio Simple": lambda tr: f_promedio_simple(tr),
    "Promedio Móvil (k=3)": lambda tr: f_promedio_movil(tr),
    "Promedio Móvil Ponderado": lambda tr: f_promedio_movil_ponderado(tr),
    "SES (α=0.1)": lambda tr: f_ses(tr),
    "Regresión Lineal": lambda tr: f_regresion_lineal(tr),
    "Holt (α=0.1, β=0.1)": lambda tr: f_holt(tr),
    "Holt-Winters (α=β=γ=0.1)": lambda tr: f_holt_winters(tr),
    "ARIMA(1,1,1)": lambda tr: f_arima(tr),
    "SARIMA(1,1,1)(1,1,1)[12]": lambda tr: f_sarima(tr),
}

# Métodos que producen pronóstico esencialmente constante
METODOS_CONST = {
    "Promedio Simple",
    "Promedio Móvil (k=3)",
    "Promedio Móvil Ponderado",
}

# -------------------------------------------------
# Walk-forward 1 paso
# -------------------------------------------------
def walk_forward_errors(series: pd.Series, min_train: int = 6):
    y_vals = series.dropna().values.tolist()
    n = len(y_vals)
    results = {name: {"y_true": [], "y_pred": []} for name in METHODS}

    for t in range(min_train, n):
        train = y_vals[:t]
        y_t = y_vals[t]
        for name, f in METHODS.items():
            try:
                yhat = f(train)
            except Exception:
                yhat = np.nan
            results[name]["y_true"].append(y_t)
            results[name]["y_pred"].append(yhat)

    rows = []
    for name, d in results.items():
        yt, yp = np.array(d["y_true"], dtype=float), np.array(d["y_pred"], dtype=float)
        mask = ~np.isnan(yp)
        if mask.sum() == 0:
            rows.append({"Método": name, "MSE": np.nan, "MAD": np.nan, "MAPE": np.nan, "n_preds": 0})
            continue
        rows.append({
            "Método": name,
            "MSE": mse(yt[mask], yp[mask]),
            "MAD": mad(yt[mask], yp[mask]),
            "MAPE": mape(yt[mask], yp[mask]),
            "n_preds": int(mask.sum())
        })
    df_err = pd.DataFrame(rows).sort_values(["MAPE", "MSE", "MAD"])
    return df_err

# agregar grafico de errores del TOP 3 DE METODOS SELECCIONADOS
def walk_forward_detail(series: pd.Series, min_train: int = 6):
    """
    Devuelve, para CADA método, las listas de:
      - y_true (valores reales en cada t)
      - y_pred (pronóstico 1 paso para ese t)
      - dates (fechas correspondientes)
    Se usa para graficar la serie de errores por método.
    """
    s = series.dropna()
    y_vals = s.values
    idx = s.index
    n = len(y_vals)

    results = {name: {"y_true": [], "y_pred": [], "dates": []} for name in METHODS}

    for t in range(min_train, n):
        train = y_vals[:t]
        y_t = y_vals[t]
        date_t = idx[t]
        for name, f in METHODS.items():
            try:
                yhat = f(train)
            except Exception:
                yhat = np.nan
            results[name]["y_true"].append(float(y_t))
            results[name]["y_pred"].append(float(yhat) if yhat is not None else np.nan)
            results[name]["dates"].append(date_t)

    return results


# -------------------------------------------------
# Módulo 2 — Clasificación + Evaluación
# -------------------------------------------------
@app.callback(
    Output("mod2-output", "children"),
    Input("upload-data", "contents")
)
def ejecutar_modulo2(contents):
    if contents is None:
        return html.Div("📁 Sube un archivo Excel para analizar la serie.", className="text-muted")

    try:
        serie = cargar_serie(contents)
        if len(serie.dropna()) < 24:
            return dbc.Alert("⚠️ No se puede ejecutar la clasificación ni los pronósticos: se requieren al menos 24 meses.", color="warning")

        # Clasificación
        clasif = clasificar_serie(serie)
        card_clasif = dbc.Card([
            dbc.CardHeader("📊 Clasificación de la serie"),
            dbc.CardBody([
                html.P(f"Tendencia: {clasif['tendencia']} (p={clasif['p_tendencia']})"),
                html.P(f"Estacionalidad: {clasif['estacionalidad']} (autocorr12={clasif['autocorr_12']})"),
                html.P(f"Estacionaria: {clasif['estacionaria']} (pADF={clasif['p_adf']})"),
            ])
        ], className="mb-4")

        # Evaluación métodos
        df_metrics = walk_forward_errors(serie, min_train=6)

        # Regla: solo permitir métodos constantes si la serie es MUY lineal
        permitir_constantes = es_muy_lineal(serie)
        nota_filtro = None
        if not permitir_constantes:
            df_metrics = df_metrics[~df_metrics["Método"].isin(METODOS_CONST)].reset_index(drop=True)
            nota_filtro = dbc.Alert(
                "ℹ️ Como la serie no es fuertemente lineal. Se excluyeron del ranking: "
                "Promedio Simple, Promedio Móvil y Promedio Móvil Ponderado.",
                color="info"
            )

        if df_metrics.empty:
            return dbc.Alert("❌ No hay métodos elegibles tras aplicar las reglas de selección.", color="danger")

        best = df_metrics.head(1)["Método"].iloc[0]


        tabla = dash_table.DataTable(
            data=df_metrics.round(3).to_dict("records"),
            columns=[{"name": i, "id": i} for i in df_metrics.columns],
            style_data_conditional=[
                {"if": {"filter_query": f'{{Método}} = "{best}"'},
                 "backgroundColor": "#d4edda", "fontWeight": "bold"}
            ],
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            page_size=9
        )

        # ====== Nuevo: gráfico de comparación de errores del Top-3 ======
        # Tomamos el Top-3 tras el filtrado aplicado
        top_methods = df_metrics["Método"].head(3).tolist()

        # Obtenemos el detalle de y_true/y_pred por método
        detail = walk_forward_detail(serie, min_train=6)

        # Construimos la figura
        fig_err = go.Figure()
        fig_err.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.35)", line_width=1)

        any_curve = False
        for m in top_methods:
            if m not in detail:
                continue
            y_true = np.array(detail[m]["y_true"], dtype=float)
            y_pred = np.array(detail[m]["y_pred"], dtype=float)
            dates  = detail[m]["dates"]

            mask = ~np.isnan(y_pred)
            if mask.sum() == 0:
                continue

            errs = y_true[mask] - y_pred[mask]
            dates_masked = [dates[i] for i, keep in enumerate(mask) if keep]

            fig_err.add_trace(go.Scatter(
                x=dates_masked,
                y=errs,
                mode="lines",
                name=f"Errores — {m}"
            ))
            any_curve = True

        fig_err.update_layout(
            title="Comparación de errores (Demanda − Pronóstico) — Top-3 métodos",
            xaxis_title="Fecha",
            yaxis_title="Error",
            margin=dict(l=10, r=10, t=60, b=10),
            height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        grafico_errores = dcc.Graph(figure=fig_err, style={"height": "340px"}) if any_curve else html.Div(
            "No fue posible calcular la serie de errores para los métodos seleccionados.", className="text-muted"
        )
        # ====== Fin nuevo ======

        return html.Div([
            card_clasif,
            html.H5("📈 Evaluación de métodos:"),
            tabla,
            html.Br(),
            nota_filtro if nota_filtro else html.Div(),
            dbc.Alert(f"✅ Mejor método por MAPE: {best}", color="success"),
            html.Hr(),
            html.H5("📉 Comparación de errores del Top-3"),
            grafico_errores
        ])

    except Exception as e:
        return dbc.Alert(f"❌ Error en el análisis: {e}", color="danger")

# -------------------------------------------------
# Helpers para optimización de parámetros (Módulo 3)
# -------------------------------------------------
def _make_predictor(method_name: str, params: dict, m_season: int = 12):
    """Devuelve una función f_pred(train) que pronostica 1 paso con los 'params' dados."""
    m = method_name.lower()
    params = params or {}

    if "ses" in m:
        a = float(params.get("alpha", 0.1))
        def f_pred(train):
            model = SimpleExpSmoothing(train, initialization_method="heuristic").fit(
                smoothing_level=a, optimized=False
            )
            return float(model.forecast(1)[0])
        return f_pred

    if "holt-winters" in m:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        g = float(params.get("gamma", 0.1))
        def f_pred(train):
            if len(train) < m_season + 2:
                model = ExponentialSmoothing(train, trend="add", initialization_method="heuristic")
                fit = model.fit(smoothing_level=a, smoothing_trend=b, optimized=False)
            else:
                model = ExponentialSmoothing(
                    train, trend="add", seasonal="add",
                    seasonal_periods=m_season, initialization_method="heuristic"
                )
                fit = model.fit(
                    smoothing_level=a, smoothing_trend=b, smoothing_seasonal=g, optimized=False
                )
            return float(fit.forecast(1)[0])
        return f_pred

    if "holt" in m and "winters" not in m:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        def f_pred(train):
            model = ExponentialSmoothing(train, trend="add", initialization_method="heuristic")
            fit = model.fit(smoothing_level=a, smoothing_trend=b, optimized=False)
            return float(fit.forecast(1)[0])
        return f_pred

    if "sarima" in m:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        P = int(params.get("P", 1)); D = int(params.get("D", 1)); Q = int(params.get("Q", 1))
        def f_pred(train):
            if len(train) < m_season + 2:
                mod = SARIMAX(train, order=(p, d, q),
                              enforce_stationarity=False, enforce_invertibility=False)
            else:
                mod = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m_season),
                              enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            return float(res.forecast(1)[0])
        return f_pred

    if "arima" in m:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        def f_pred(train):
            if len(train) < 6:
                return float(train[-1])
            mod = SARIMAX(train, order=(p, d, q),
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            return float(res.forecast(1)[0])
        return f_pred

    if "regresión lineal" in m:
        def f_pred(train):
            x = np.arange(len(train))
            slope, intercept, *_ = stats.linregress(x, train)
            return float(intercept + slope * len(train))
        return f_pred

    def f_pred(train):  # fallback
        return float(train[-1])
    return f_pred

def _arange_005(lo=0.05, hi=0.95, step=0.05):
    """Arreglo [lo, hi] con paso 0.05 (incluye extremos si calzan)."""
    vals = np.round(np.arange(lo, hi + 1e-9, step), 2)
    vals = vals[(vals >= lo) & (vals <= hi)]
    return list(vals)

def _grid(method_name: str):
    """Parrillas compactas por método."""
    m = method_name.lower()

    if "ses" in m:
        alphas = _arange_005()
        return [{"alpha": a} for a in alphas]

    if "holt" in m and "winters" not in m:
        alphas = _arange_005()
        betas  = _arange_005()
        return [{"alpha": a, "beta": b} for a in alphas for b in betas]

    if "holt-winters" in m:
        # Fase 1 (gruesa)
        coarse = [0.1, 0.3, 0.5, 0.7, 0.9]
        return [{"alpha": a, "beta": b, "gamma": g} for a in coarse for b in coarse for g in coarse]

    if "sarima" in m:
        ps = [0,1,2]; ds = [0,1]; qs = [0,1,2]
        Ps = [0,1];  Ds = [0,1];  Qs = [0,1]
        out = []
        for p in ps:
            for d in ds:
                for q in qs:
                    for P in Ps:
                        for D in Ds:
                            for Q in Qs:
                                out.append({"p": p,"d": d,"q": q,"P": P,"D": D,"Q": Q,"s": 12})
        return out

    if "arima" in m:
        ps = [0,1,2]; ds = [0,1]; qs = [0,1,2]
        return [{"p": p, "d": d, "q": q} for p in ps for d in ds for q in qs]

    return [{}]

def _refine_around(best_params: dict, lo=0.05, hi=0.95, step=0.05):
    """Genera parrilla fina alrededor de best_params (±0.10) con paso 0.05, acotando [0.05, 0.95]."""
    def clamp(v): return float(np.round(min(max(v, lo), hi), 2))
    alphas = betas = gammas = None
    if "alpha" in best_params:
        a = best_params["alpha"]
        alphas = [clamp(a + d) for d in [-0.10, -0.05, 0, 0.05, 0.10]]
    if "beta" in best_params:
        b = best_params["beta"]
        betas = [clamp(b + d) for d in [-0.10, -0.05, 0, 0.05, 0.10]]
    if "gamma" in best_params:
        g = best_params["gamma"]
        gammas = [clamp(g + d) for d in [-0.10, -0.05, 0, 0.05, 0.10]]

    fine = []
    for a in (alphas or [None]):
        for b in (betas or [None]):
            for g in (gammas or [None]):
                p = {}
                if a is not None: p["alpha"] = a
                if b is not None: p["beta"] = b
                if g is not None: p["gamma"] = g
                fine.append(p)
    # dedup
    seen, uniq = set(), []
    for d in fine:
        t = tuple(sorted(d.items()))
        if t not in seen:
            seen.add(t); uniq.append(d)
    return uniq

def _eval_walkforward(vals, f_pred, min_train=6):
    """Evalúa MAPE con walk-forward 1 paso."""
    preds, reals = [], []
    n = len(vals)
    for t in range(min_train, n):
        train = vals[:t]
        y_t = vals[t]
        try:
            yhat = f_pred(train)
        except Exception:
            yhat = np.nan
        preds.append(yhat)
        reals.append(y_t)
    preds = np.array(preds, dtype=float)
    reals = np.array(reals, dtype=float)
    mask = ~np.isnan(preds)
    if mask.sum() == 0:
        return np.inf
    return mape(reals[mask], preds[mask])

def _tunar_mejor(serie: pd.Series, method_name: str, min_train: int = 6, m_season: int = 12,
                 mape_min_improve: float = 0.05, max_hw_combos: int = 300):
    """
    Barre parrilla y devuelve (mejores_params, mejor_mape).
    - SES/Holt: paso 0.05 completo.
    - HW: coarse-to-fine con cap de combos y early stop por mejora mínima.
    """
    vals = serie.dropna().values
    base_grid = _grid(method_name)

    # Métodos sin hiperparámetros
    if base_grid == [{}]:
        f_pred = _make_predictor(method_name, {}, m_season=m_season)
        return None, _eval_walkforward(vals, f_pred, min_train=min_train)

    # Holt-Winters: coarse → fine
    if "holt-winters" in method_name.lower():
        best_params, best_mape = None, np.inf
        combos_used = 0

        # Fase 1 (gruesa)
        for params in base_grid:
            if combos_used >= max_hw_combos:
                break
            try:
                f_pred = _make_predictor(method_name, params, m_season=m_season)
                mape_val = _eval_walkforward(vals, f_pred, min_train=min_train)
                combos_used += 1
                if mape_val < best_mape - mape_min_improve:
                    best_mape = mape_val
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            return None, np.inf

        # Fase 2 (fina alrededor del mejor, paso 0.05)
        fine_grid = _refine_around(best_params)
        for params in fine_grid:
            if combos_used >= max_hw_combos:
                break
            try:
                f_pred = _make_predictor(method_name, params, m_season=m_season)
                mape_val = _eval_walkforward(vals, f_pred, min_train=min_train)
                combos_used += 1
                if mape_val < best_mape:
                    best_mape = mape_val
                    best_params = params
            except Exception:
                continue

        return best_params, best_mape

    # Resto (SES/Holt/ARIMA/SARIMA): parrillas compactas
    best_params, best_mape = None, np.inf
    for params in base_grid:
        try:
            f_pred = _make_predictor(method_name, params, m_season=m_season)
            mape_val = _eval_walkforward(vals, f_pred, min_train=min_train)
            if mape_val < best_mape:
                best_mape = mape_val
                best_params = params
        except Exception:
            continue
    return best_params, best_mape

def _normalize_params(best_method: str, params):
    """
    Devuelve SIEMPRE un dict con tipos correctos (para guardar en best-model-store).
    """
    m = (best_method or "").lower()
    if params is None:
        params = {}
    if isinstance(params, (list, tuple, np.ndarray)):
        params = {"weights": list(map(float, params))}
    if not isinstance(params, dict):
        params = {"value": params}

    if "promedio móvil (k" in m:
        k = int(params.get("k", params.get("value", 3)))
        return {"k": k}
    elif "ponderado" in m:
        w = params.get("weights", params.get("value", [0.5, 0.3, 0.2]))
        return {"weights": list(map(float, w))}
    elif "ses" in m:
        a = float(params.get("alpha", params.get("value", 0.1)))
        return {"alpha": a}
    elif "holt-winters" in m:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        g = float(params.get("gamma", 0.1))
        return {"alpha": a, "beta": b, "gamma": g}
    elif "holt" in m and "winters" not in m:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        return {"alpha": a, "beta": b}
    elif "sarima" in m:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        P = int(params.get("P", 1)); D = int(params.get("D", 1)); Q = int(params.get("Q", 1))
        s = int(params.get("s", 12))
        return {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s}
    elif "arima" in m:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        return {"p": p, "d": d, "q": q}
    else:
        return {}

# -------------------------------------------------
# Módulo 3 — Optimización del método ganador
# -------------------------------------------------
@app.callback(
    [Output("mod3-output", "children"),
     Output("best-model-store", "data")],
    Input("mod2-output", "children"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def optimizar_mejor_modelo(_mod2_ready, contents):
    if contents is None or _mod2_ready is None:
        raise PreventUpdate

    try:
        serie = cargar_serie(contents)
        if len(serie.dropna()) < 24:
            return dbc.Alert("⚠️ La serie no tiene suficientes datos para optimizar parámetros.", color="warning"), None

        # Reevaluar para obtener ganador actual (misma lógica del Mód 2)
        df_metrics = walk_forward_errors(serie, min_train=6)

        # Regla: solo permitir métodos constantes si la serie es MUY lineal
        permitir_constantes = es_muy_lineal(serie)
        if not permitir_constantes:
            df_metrics = df_metrics[~df_metrics["Método"].isin(METODOS_CONST)].reset_index(drop=True)

        if df_metrics.empty:
            return dbc.Alert("❌ No hay métodos elegibles para optimizar tras aplicar las reglas de selección.", color="danger"), None

        best_method = df_metrics.head(1)["Método"].iloc[0]
        base_mape = float(df_metrics.head(1)["MAPE"].iloc[0])

        # m estacional (constante por ahora)
        m_sugerido = 12

        # 🔧 Barrido de hiperparámetros del método ganador
        mejor_params, mejor_mape = _tunar_mejor(
            serie, best_method, min_train=6, m_season=m_sugerido,
            mape_min_improve=0.05,  # 0.05 pp de mejora mínima en HW (early-stop)
            max_hw_combos=300       # cap de combinaciones en HW
        )

        msgs = [
            html.P(f"Método inicial: {best_method} (MAPE base = {base_mape:.2f}%)"),
        ]
        if mejor_params is not None and len(mejor_params) > 0:
            msgs.append(html.P(f"Mejores parámetros: {mejor_params}"))

        if np.isfinite(mejor_mape):
            delta = base_mape - mejor_mape
            signo = "↓" if delta > 0 else "→"
            msgs.append(html.P(f"MAPE optimizado = {mejor_mape:.2f}% ({signo} {abs(delta):.2f} pp)"))
        else:
            mejor_mape = base_mape
            msgs.append(html.P("No fue posible mejorar el MAPE con la parrilla evaluada."))

        best_payload = {
            "method": best_method,
            "params": _normalize_params(best_method, mejor_params)
        }
        return html.Div(msgs), best_payload

    except Exception as e:
        return dbc.Alert(f"❌ Error durante la optimización: {e}", color="danger"), None

# -------------------------------------------------
# Módulo 4 — Pronóstico
# -------------------------------------------------
def _fit_and_forecast(series: pd.Series, method: str, params: dict, H: int) -> pd.Series:
    """Ajusta el método ganador a toda la serie y retorna un forecast de H pasos."""
    vals = series.dropna().values
    method = method or ""
    params = params or {}

    # Fechas futuras
    start = series.index[-1] + MonthBegin(1)
    future_idx = pd.date_range(start, periods=H, freq="MS")

    # ----- Métodos simples -----
    if method == "Promedio Simple":
        yhat = float(np.mean(vals))
        return pd.Series([yhat]*H, index=future_idx)

    if "Promedio Móvil (k=3)" in method:
        k = int(params.get("k", 3))
        base = float(np.mean(vals[-k:])) if len(vals) >= k else float(np.mean(vals))
        return pd.Series([base]*H, index=future_idx)

    if "Ponderado" in method:
        w = np.array(params.get("weights", [0.5, 0.3, 0.2]), dtype=float)[::-1]
        k = len(w)
        base = float(np.dot(vals[-k:], w)) if len(vals) >= k else float(np.mean(vals))
        return pd.Series([base]*H, index=future_idx)

    if "Regresión Lineal" in method:
        x = np.arange(len(vals))
        slope, intercept, *_ = stats.linregress(x, vals)
        y_fc = [float(intercept + slope * (len(vals) + h)) for h in range(1, H+1)]
        return pd.Series(y_fc, index=future_idx)

    # ----- Suavizamiento exponencial -----
    if "SES" in method:
        a = float(params.get("alpha", 0.1))
        model = SimpleExpSmoothing(vals, initialization_method="heuristic").fit(smoothing_level=a, optimized=False)
        fc = model.forecast(H)
        return pd.Series(np.asarray(fc, dtype=float), index=future_idx)

    if "Holt" in method and "Winters" not in method:
        a = float(params.get("alpha", 0.1)); b = float(params.get("beta", 0.1))
        model = ExponentialSmoothing(vals, trend="add", initialization_method="heuristic")
        fit = model.fit(smoothing_level=a, smoothing_trend=b, optimized=False)
        fc = fit.forecast(H)
        return pd.Series(np.asarray(fc, dtype=float), index=future_idx)

    if "Holt-Winters" in method:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        g = float(params.get("gamma", 0.1))
        m = 12  # periodo estacional mensual

        # Si hay pocos datos, usa solo Holt (tendencia) para evitar errores
        if len(vals) < m + 2:   # puedes subir a 2*m si quieres ser más conservadora
            model = ExponentialSmoothing(
                vals,
                trend="add",
                initialization_method="heuristic"
            )
            fit = model.fit(
                smoothing_level=a,
                smoothing_trend=b,
                optimized=False
            )
        else:
            model = ExponentialSmoothing(
                vals,
                trend="add",
                seasonal="add",
                seasonal_periods=m,
                initialization_method="heuristic"
            )
            fit = model.fit(
                smoothing_level=a,
                smoothing_trend=b,
                smoothing_seasonal=g,
                optimized=False
            )

        fc = fit.forecast(H)
        return pd.Series(np.asarray(fc, dtype=float), index=future_idx)


    # ----- ARIMA/SARIMA -----
    if "SARIMA" in method:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        P = int(params.get("P", 1)); D = int(params.get("D", 1)); Q = int(params.get("Q", 1))
        mod = SARIMAX(vals, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fc = res.forecast(H)
        return pd.Series(np.asarray(fc, dtype=float), index=future_idx)

    if "ARIMA" in method:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        mod = SARIMAX(vals, order=(p, d, q),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fc = res.forecast(H)
        return pd.Series(np.asarray(fc, dtype=float), index=future_idx)

    # Fallback: naïve
    return pd.Series([float(vals[-1])]*H, index=future_idx)

def _fitted_series(series: pd.Series, method: str, params: dict, m_season: int = 12) -> pd.Series:
    """
    Devuelve la serie de valores AJUSTADOS (in-sample) del método ganador,
    usando TODA la historia disponible.

    Es lo que se grafica como "pronóstico ajustado sobre histórico".
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    serie = series.dropna()
    vals = serie.values
    idx = serie.index
    method = method or ""
    params = params or {}

    # ----- Métodos simples -----
    if method == "Promedio Simple":
        yhat = float(np.mean(vals))
        return pd.Series([yhat] * len(vals), index=idx)

    if "Promedio Móvil (k=3)" in method:
        k = int(params.get("k", 3))
        s = pd.Series(vals, index=idx).rolling(k).mean()
        return s

    if "Ponderado" in method:
        w = np.array(params.get("weights", [0.5, 0.3, 0.2]), dtype=float)
        k = len(w)
        s = pd.Series(vals, index=idx)
        fitted = s.rolling(k).apply(lambda x: float(np.dot(x.values, w[::-1])), raw=False)
        return fitted

    if "Regresión Lineal" in method:
        x = np.arange(len(vals))
        slope, intercept, *_ = stats.linregress(x, vals)
        fitted = intercept + slope * x
        return pd.Series(fitted, index=idx)

    # ----- Suavizamiento exponencial -----
    if "SES" in method:
        a = float(params.get("alpha", 0.1))
        model = SimpleExpSmoothing(vals, initialization_method="heuristic").fit(
            smoothing_level=a, optimized=False
        )
        fitted = model.fittedvalues
        return pd.Series(np.asarray(fitted, dtype=float), index=idx)

    if "Holt" in method and "Winters" not in method:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        model = ExponentialSmoothing(vals, trend="add", initialization_method="heuristic")
        fit = model.fit(smoothing_level=a, smoothing_trend=b, optimized=False)
        fitted = fit.fittedvalues
        return pd.Series(np.asarray(fitted, dtype=float), index=idx)

    if "Holt-Winters" in method:
        a = float(params.get("alpha", 0.1))
        b = float(params.get("beta", 0.1))
        g = float(params.get("gamma", 0.1))
        m = m_season

        if len(vals) < m + 2:
            # Con pocos datos, solo tendencia
            model = ExponentialSmoothing(vals, trend="add", initialization_method="heuristic")
            fit = model.fit(smoothing_level=a, smoothing_trend=b, optimized=False)
        else:
            model = ExponentialSmoothing(
                vals,
                trend="add",
                seasonal="add",
                seasonal_periods=m,
                initialization_method="heuristic"
            )
            fit = model.fit(
                smoothing_level=a,
                smoothing_trend=b,
                smoothing_seasonal=g,
                optimized=False
            )
        fitted = fit.fittedvalues
        return pd.Series(np.asarray(fitted, dtype=float), index=idx)

    # ----- ARIMA / SARIMA -----
    if "SARIMA" in method:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        P = int(params.get("P", 1)); D = int(params.get("D", 1)); Q = int(params.get("Q", 1))
        mod = SARIMAX(vals, order=(p, d, q), seasonal_order=(P, D, Q, m_season),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fitted = res.get_prediction(start=0, end=len(vals)-1).predicted_mean
        return pd.Series(np.asarray(fitted, dtype=float), index=idx)

    if "ARIMA" in method:
        p = int(params.get("p", 1)); d = int(params.get("d", 1)); q = int(params.get("q", 1))
        mod = SARIMAX(vals, order=(p, d, q),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fitted = res.get_prediction(start=0, end=len(vals)-1).predicted_mean
        return pd.Series(np.asarray(fitted, dtype=float), index=idx)

    # Fallback: devuelve la propia serie (naïve)
    return serie.copy()



def _backtest_with_best(series, method: str, params: dict, min_train: int = 10, m_season: int = 12) -> pd.Series:
    """
    Genera una serie de pronósticos 1 paso adelante sobre toda la historia,
    usando SIEMPRE el método ganador y sus parámetros optimizados.

    - Usa _make_predictor, igual que en la optimización.
    - No mete “trucos” de fallback plano; donde el modelo falla deja NaN.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    serie = series.dropna()
    vals = serie.values
    idx = serie.index
    n = len(vals)

    # Preparar array de predicciones (NaN al inicio)
    preds = np.full(n, np.nan, dtype=float)

    # Construir el predictor de 1 paso con el método ganador
    f_pred = _make_predictor(method, params or {}, m_season=m_season)

    for t in range(min_train, n):
        train_vals = vals[:t]
        try:
            yhat = f_pred(train_vals)
        except Exception:
            yhat = np.nan
        preds[t] = yhat

    return pd.Series(preds, index=idx)



@app.callback(
    [Output("forecast-plot", "children"),
     Output("forecast-table", "children")],
    [Input("horizon-select", "value"),
     Input("best-model-store", "data")],
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def render_forecast(H, best_payload, contents):
    if contents is None or best_payload is None:
        raise PreventUpdate

    try:
        # Serie histórica
        serie = cargar_serie(contents).dropna()
        if len(serie) < 24:
            return (
                dbc.Alert("⚠️ Se requieren al menos 24 meses para pronosticar.", color="warning"),
                html.Div()
            )

        method = (best_payload or {}).get("method")
        params = (best_payload or {}).get("params", {})

        # 1) Pronóstico AJUSTADO sobre la historia (in-sample)
        fitted = _fitted_series(serie, method, params, m_season=12)

        # Cálculo de MAPE in-sample (donde haya valores válidos)
        mask_hist = ~np.isnan(fitted.values)
        if mask_hist.sum() > 0:
            mape_hist = mape(serie.values[mask_hist], fitted.values[mask_hist])
            subtitle = f"(MAPE histórico ajustado ≈ {mape_hist:.2f}%)"
        else:
            subtitle = "(No fue posible calcular pronóstico ajustado sobre histórico)"

        # 2) Pronóstico futuro con el método ganador y sus parámetros
        fc = _fit_and_forecast(serie, method, params, int(H))
        if not isinstance(fc, pd.Series):
            fc = pd.Series(fc)
        # --- unir histórico y futuro ---
        last_date = serie.index[-1]
        last_val  = float(serie.iloc[-1])

        # Construimos un pronóstico "extendido" que inicia en el último punto observado
        fc_x = [last_date] + list(fc.index)     # fechas: último histórico + horizontes
        fc_y = [last_val]  + list(fc.values)    # valores: último real + pronóstico


        # 3) Gráfico: histórico, ajustado y futuro
        fig = go.Figure()

        # línea vertical que marca el corte historia→futuro
        fig.add_vline(x=last_date, line_dash="dot", line_width=1, line_color="rgba(0,0,0,0.25)")


        # Demanda real (línea azul con puntos)
        fig.add_trace(go.Scatter(
            x=serie.index,
            y=serie.values,
            mode="lines+markers",  # 🔹 agrega los puntos
            name="Demanda (histórico)",
            marker=dict(size=5, symbol="circle"),
            line=dict(color="royalblue")
        ))

        # Pronóstico AJUSTADO sobre la historia
        if mask_hist.sum() > 0:
            fig.add_trace(go.Scatter(
                x=fitted.index,
                y=fitted.values,
                mode="lines",
                name="Pronóstico ajustado sobre histórico",
                line=dict(dash="dot")
            ))

        # Pronóstico futuro (continuo desde el último punto histórico)
        fig.add_trace(go.Scatter(
            x=fc_x,
            y=fc_y,
            mode="lines+markers",
            name=f"Pronóstico futuro (+{H} meses)"
        ))

        fig.update_layout(
            title=f"Pronóstico con método ganador — {method} {subtitle}",
            xaxis_title="Fecha",
            yaxis_title="Demanda",
            margin=dict(l=10, r=10, t=70, b=10),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        graph = dcc.Graph(figure=fig, style={"height": "420px"})

        # Tabla de pronóstico futuro (solo los H meses hacia adelante)
        df_fc = pd.DataFrame({
            "Fecha": fc.index.strftime("%Y-%m"),
            "Pronóstico": np.round(fc.values, 3)
        })

        table = dash_table.DataTable(
            data=df_fc.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_fc.columns],
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
            page_size=min(len(df_fc), 12),
            style_table={'maxWidth': '520px'}
        )

        return graph, table

    except Exception as e:
        return (
            dbc.Alert(f"❌ Error al pronosticar: {e}", color="danger"),
            html.Div()
        )


from dash import dcc, no_update

@app.callback(
    Output("download-forecast", "data"),
    Input("btn-download-forecast", "n_clicks"),
    State("best-model-store", "data"),
    State("horizon-select", "value"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def descargar_pronostico(n_clicks, best_payload, H, contents):
    if not n_clicks:
        raise PreventUpdate
    if contents is None or best_payload is None or H is None:
        return no_update

    try:
        # 1) Serie histórica
        serie = cargar_serie(contents).dropna()
        if len(serie) < 24:
            return no_update

        # 2) Método ganador + params
        method = (best_payload or {}).get("method")
        params = (best_payload or {}).get("params", {})
        if not method:
            return no_update

        # 3) Pronóstico futuro (H)
        fc = _fit_and_forecast(serie, method, params, int(H))
        if not isinstance(fc, pd.Series):
            fc = pd.Series(fc)

        # 4) DataFrame con 3 columnas
        df_out = pd.DataFrame({
            "year":  fc.index.year.astype(int),
            "month":  [MESES_ES[m] for m in fc.index.month],
            "forecast": np.round(fc.values, 3)
        })

        # 5) Intentar Excel; si falla (sin openpyxl), caer a CSV
        try:
            return dcc.send_data_frame(
                df_out.to_excel, f"forecast_{H}m.xlsx",
                sheet_name="forecast", index=False
            )
        except Exception:
            # Fallback robusto: CSV UTF-8 con BOM (abre bien en Excel)
            return dcc.send_data_frame(
                df_out.to_csv, f"forecast_{H}m.csv",
                index=False, encoding="utf-8-sig"
            )

    except Exception as e:
        # Si algo raro ocurre, no disparamos descarga
        # (opcional) print(e) para ver el error en consola
        return no_update


# -------------------------------------------------
# Run (Dashhh)
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

