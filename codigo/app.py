# ======================================================
# app.py - Motor de Pronosticos Tuboplex (interfaz Dash)
# ======================================================
#
# Capa DELGADA. Toda la logica de dominio vive en forecasting_core, que no
# importa dash ni plotly y es testeable de forma independiente (Fase 0).
# Este archivo solo construye el layout y conecta callbacks; no recalcula
# metricas ni implementa modelos.
#
# Resuelve F16: sin warnings.filterwarnings("ignore") global. Los
# ConvergenceWarning de statsmodels quedan en el log, con nivel configurable
# via la variable de entorno FORECASTING_LOG_LEVEL (por defecto WARNING).
from __future__ import annotations

import base64
import io
import logging
import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

from forecasting_core.data import load_series
from forecasting_core.intervals import prediction_interval
from forecasting_core.inventory import compute_policy
from forecasting_core.models import get_spec
from forecasting_core.optimize import run_pipeline

logging.basicConfig(
    level=os.environ.get("FORECASTING_LOG_LEVEL", "WARNING"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("forecasting_app")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Motor de Pronosticos - Tuboplex"
server = app.server


# -------------------------------------------------
# Carga de archivo (Dash entrega base64; el parseo real vive en forecasting_core.data)
# -------------------------------------------------
def _decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def cargar_serie_desde_upload(contents: str, gap_policy: str = "report"):
    decoded = _decode_upload(contents)
    df = pd.read_excel(io.BytesIO(decoded), dtype=str)
    return load_series(df, gap_policy=gap_policy)


# -------------------------------------------------
# Layout
# -------------------------------------------------
app.layout = dbc.Container([
    html.H2("1. Carga y validacion de datos", className="mt-3 mb-4"),

    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Arrastra o selecciona tu archivo Excel con columnas ",
            html.Code("year, month, demand"),
            " (columna opcional ", html.Code("sku"), " para varios productos)",
        ]),
        style={
            "width": "100%", "height": "90px", "lineHeight": "90px",
            "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "10px",
            "textAlign": "center", "margin": "10px",
        },
        multiple=False,
    ),

    dbc.Row([
        dbc.Col([
            html.Label("Politica ante meses faltantes:"),
            dcc.RadioItems(
                id="gap-policy",
                options=[
                    {"label": " Reportar (dejar visible, no imputar)", "value": "report"},
                    {"label": " Interpolar", "value": "interpolate"},
                    {"label": " Rellenar con cero", "value": "zero"},
                ],
                value="report", inline=False,
            ),
        ], width="auto"),
    ], className="mb-3"),

    html.Div(id="alerta", className="mt-2"),
    html.Div(id="reporte-carga", className="mt-2"),
    html.Div(id="preview", className="mt-3"),

    html.Hr(),
    html.H4("2. Clasificacion estructural y evaluacion honesta de metodos", className="mt-2"),
    html.P(
        "La metrica de ranking es MASE (escala-independiente, definida en cero). "
        "El bloque final de evaluacion nunca participo en la eleccion de hiperparametros "
        "(ver metodologia, Fase 3 del refactor).",
        className="text-muted small",
    ),
    dcc.Loading(html.Div(id="mod2-output", className="mt-3"), type="default"),

    html.Hr(),
    html.H4("3. Pronostico del metodo ganador, con intervalos", className="mt-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Horizonte del forecast:"),
            dcc.RadioItems(
                id="horizon-select",
                options=[{"label": " {} meses".format(h), "value": h}
                         for h in (6, 12, 18, 24)],
                value=12, inline=True,
            ),
        ], width="auto"),
        dbc.Col([
            html.Label("Nivel del intervalo de prediccion:"),
            dcc.RadioItems(
                id="level-select",
                options=[{"label": " {:.0%}".format(v), "value": v}
                         for v in (0.80, 0.90, 0.95)],
                value=0.95, inline=True,
            ),
        ], width="auto"),
    ], className="mb-3"),

    dcc.Loading(html.Div(id="forecast-plot", className="mt-2"), type="default"),
    dcc.Loading(html.Div(id="forecast-table", className="mt-3"), type="default"),

    dbc.Button("Descargar forecast (Excel)", id="btn-download-forecast",
               color="primary", className="mt-3"),
    dcc.Download(id="download-forecast"),

    html.Hr(),
    html.H4("4. Politica de inventario (stock de seguridad, punto de reorden)", className="mt-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Lead time (meses):"),
            dcc.Input(id="lead-time", type="number", value=3, min=1, max=24, step=1),
        ], width="auto"),
        dbc.Col([
            html.Label("Nivel de servicio:"),
            dcc.Dropdown(
                id="service-level",
                options=[{"label": "{:.1%}".format(v), "value": v}
                         for v in (0.90, 0.95, 0.975, 0.99)],
                value=0.95, style={"width": "140px"},
            ),
        ], width="auto"),
    ], className="mb-3"),
    dcc.Loading(html.Div(id="inventory-output", className="mt-2"), type="default"),

    dcc.Store(id="series-store"),
    dcc.Store(id="best-model-store"),
], fluid=True)


# -------------------------------------------------
# Modulo 0 - Carga y validacion
# -------------------------------------------------
@app.callback(
    [Output("alerta", "children"),
     Output("reporte-carga", "children"),
     Output("preview", "children"),
     Output("series-store", "data")],
    Input("upload-data", "contents"),
    [State("upload-data", "filename"),
     State("gap-policy", "value")],
)
def validar_y_mostrar(contents, filename, gap_policy):
    if contents is None:
        return "", "", "", None

    try:
        result = cargar_serie_desde_upload(contents, gap_policy=gap_policy or "report")
    except Exception as exc:
        logger.exception("Error al parsear el archivo subido")
        return dbc.Alert("Error al leer el archivo: {}".format(exc), color="danger"), "", "", None

    report = result.report
    if not report.ok:
        alerta = dbc.Alert(
            [html.P("No fue posible cargar la serie:", className="mb-1")]
            + [html.P("- {}".format(e), className="mb-0 small") for e in report.errors],
            color="danger",
        )
        return alerta, "", "", None

    serie = result.series
    n, inicio, fin = report.n_obs, report.start, report.end
    alerta = dbc.Alert(
        "Serie valida ({} meses) - Periodo: {:%Y-%m} -> {:%Y-%m}.".format(n, inicio, fin),
        color="success", dismissable=True,
    )

    reporte_items = []
    if report.notes:
        reporte_items.append(html.P("Notas de carga:", className="mb-1 fw-bold small"))
        reporte_items += [html.P("- {}".format(nt), className="mb-0 small text-muted")
                          for nt in report.notes]
    reporte = dbc.Alert(reporte_items, color="info") if reporte_items else ""

    df_preview = serie.reset_index()
    df_preview.columns = ["Fecha", "Demanda"]
    df_preview["Fecha"] = df_preview["Fecha"].dt.strftime("%Y-%m")
    tabla = dash_table.DataTable(
        data=df_preview.head(12).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_preview.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        page_size=12,
    )
    fig = go.Figure(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers",
                               name="Demanda"))
    fig.update_layout(title="Demanda mensual (serie cargada)", xaxis_title="Fecha",
                      yaxis_title="Demanda", margin=dict(l=10, r=10, t=50, b=10), height=340)

    preview = html.Div([dcc.Graph(figure=fig, style={"height": "340px"}), html.Br(), tabla])

    store = {"index": [d.isoformat() for d in serie.index], "values": serie.values.tolist()}
    return alerta, reporte, preview, store


def _serie_desde_store(data) -> pd.Series | None:
    if not data:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(data["index"]))
    return pd.Series(data["values"], index=idx).asfreq("MS")


# -------------------------------------------------
# Modulo 2 - Clasificacion + evaluacion honesta
# -------------------------------------------------
@app.callback(
    [Output("mod2-output", "children"),
     Output("best-model-store", "data")],
    Input("series-store", "data"),
)
def ejecutar_modulo2(data):
    serie = _serie_desde_store(data)
    if serie is None:
        return html.Div("Sube un archivo Excel para analizar la serie.", className="text-muted"), None

    try:
        resultado = run_pipeline(serie, m=12)
    except Exception as exc:
        logger.exception("Fallo el pipeline de pronostico")
        return dbc.Alert("Error en el analisis: {}".format(exc), color="danger"), None

    prof = resultado.profile
    card_clasif = dbc.Card([
        dbc.CardHeader("Clasificacion de la serie"),
        dbc.CardBody([
            html.P("Tendencia: {} (p={:.4f}) - {}".format(
                "Si" if prof.has_trend else "No", prof.trend_pvalue, prof.trend_test)),
            html.P("Estacionalidad: {} (F_S={}) - {}".format(
                "Si" if prof.has_seasonality else "No",
                "n/e" if not np.isfinite(prof.seasonal_strength) else round(prof.seasonal_strength, 3),
                prof.seasonality_test)),
            html.P("Estacionariedad: {}".format(prof.stationarity_verdict)),
        ] + ([html.P(w, className="text-warning small mb-0") for w in prof.warnings]
             if prof.warnings else [])),
    ], className="mb-4")

    if not resultado.ok:
        return html.Div([card_clasif, dbc.Alert(
            " | ".join(resultado.errors) or "No hay metodos elegibles.", color="danger")]), None

    ranking = resultado.evaluation.ranked.copy()
    ranking_display = ranking[["etiqueta", "mase", "mape", "mad", "me", "n_preds"]].rename(
        columns={"etiqueta": "Metodo", "mase": "MASE", "mape": "MAPE (%)",
                 "mad": "MAD", "me": "ME (sesgo)", "n_preds": "n"}
    ).round(3)
    best_label = ranking.iloc[0]["etiqueta"]

    tabla = dash_table.DataTable(
        data=ranking_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in ranking_display.columns],
        style_data_conditional=[{
            "if": {"filter_query": '{{Metodo}} = "{}"'.format(best_label)},
            "backgroundColor": "#d4edda", "fontWeight": "bold",
        }],
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        page_size=len(ranking_display),
    )

    excl_items = [html.P("- {}: {}".format(k, v), className="mb-0 small text-muted")
                  for k, v in resultado.excluded.items()]
    excl_card = (dbc.Alert(
        [html.P("Metodos excluidos por el filtro estructural:", className="mb-1 fw-bold small")]
        + excl_items, color="info") if excl_items else "")

    notas = [html.P(n, className="small text-muted mb-0") for n in resultado.notes]

    top3 = ranking["modelo"].head(3).tolist()
    fig_err = go.Figure()
    fig_err.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.35)", line_width=1)
    errores = resultado.evaluation.errors_frame()
    for key in top3:
        if key not in errores.columns:
            continue
        etiqueta = resultado.evaluation.metrics.set_index("modelo").loc[key, "etiqueta"]
        fig_err.add_trace(go.Scatter(x=errores.index, y=errores[key], mode="lines",
                                     name="Error - {}".format(etiqueta)))
    fig_err.update_layout(
        title="Errores fuera de muestra (bloque de evaluacion) - Top 3",
        xaxis_title="Fecha", yaxis_title="Real - Pronostico",
        margin=dict(l=10, r=10, t=60, b=10), height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    winner_key = resultado.winner
    winner_params = resultado.winner_params
    payload = {"metodo": winner_key, "params": winner_params}

    return html.Div([
        card_clasif, html.H5("Evaluacion de metodos (bloque final, fuera de muestra):"),
        tabla, html.Br(), excl_card, html.Div(notas),
        dbc.Alert("Mejor metodo por MASE: {}".format(best_label), color="success"),
        html.Hr(), html.H5("Errores fuera de muestra del Top 3"),
        dcc.Graph(figure=fig_err, style={"height": "320px"}),
    ]), payload


# -------------------------------------------------
# Modulo 3 - Pronostico con intervalos
# -------------------------------------------------
@app.callback(
    [Output("forecast-plot", "children"), Output("forecast-table", "children")],
    [Input("horizon-select", "value"), Input("level-select", "value"),
     Input("best-model-store", "data")],
    State("series-store", "data"),
    prevent_initial_call=True,
)
def render_forecast(H, level, best_payload, data):
    serie = _serie_desde_store(data)
    if serie is None or best_payload is None:
        raise PreventUpdate

    try:
        spec = get_spec(best_payload["metodo"])
        params = best_payload["params"]
        pi = prediction_interval(serie, spec, params, season_length=12,
                                 horizon=int(H), level=float(level))
    except Exception as exc:
        logger.exception("Fallo el pronostico final")
        return dbc.Alert("Error al pronosticar: {}".format(exc), color="danger"), html.Div()

    fig = go.Figure()
    fig.add_vline(x=serie.index[-1], line_dash="dot", line_width=1, line_color="rgba(0,0,0,0.25)")
    fig.add_trace(go.Scatter(x=serie.dropna().index, y=serie.dropna().values,
                             mode="lines+markers", name="Demanda (historico)",
                             line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(
        x=list(pi.index) + list(pi.index)[::-1],
        y=list(pi.upper) + list(pi.lower)[::-1],
        fill="toself", fillcolor="rgba(0,150,80,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="Intervalo {:.0%}".format(pi.level), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=pi.index, y=pi.mean, mode="lines+markers",
                             name="Pronostico futuro (+{} meses)".format(H),
                             line=dict(color="seagreen")))
    fig.update_layout(
        title="Pronostico - {} ({})".format(best_payload["metodo"], pi.method),
        xaxis_title="Fecha", yaxis_title="Demanda",
        margin=dict(l=10, r=10, t=70, b=10), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    graph = dcc.Graph(figure=fig, style={"height": "420px"})

    df_fc = pi.to_frame().reset_index(names="Fecha")
    df_fc["Fecha"] = df_fc["Fecha"].dt.strftime("%Y-%m")
    table = dash_table.DataTable(
        data=df_fc.round(2).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_fc.columns],
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        page_size=min(len(df_fc), 12),
    )
    return graph, table


@app.callback(
    Output("download-forecast", "data"),
    Input("btn-download-forecast", "n_clicks"),
    [State("best-model-store", "data"), State("horizon-select", "value"),
     State("level-select", "value"), State("series-store", "data")],
    prevent_initial_call=True,
)
def descargar_pronostico(n_clicks, best_payload, H, level, data):
    if not n_clicks:
        raise PreventUpdate
    serie = _serie_desde_store(data)
    if serie is None or best_payload is None:
        return no_update
    try:
        spec = get_spec(best_payload["metodo"])
        pi = prediction_interval(serie, spec, best_payload["params"], season_length=12,
                                 horizon=int(H), level=float(level))
        df_out = pi.to_frame().reset_index(names="fecha")
        df_out.insert(1, "metodo", best_payload["metodo"])
        return dcc.send_data_frame(df_out.to_excel, "forecast_{}m.xlsx".format(H),
                                   sheet_name="forecast", index=False)
    except Exception as exc:
        logger.exception("Fallo la descarga")
        return no_update


# -------------------------------------------------
# Modulo 4 - Inventario
# -------------------------------------------------
@app.callback(
    Output("inventory-output", "children"),
    [Input("lead-time", "value"), Input("service-level", "value"),
     Input("best-model-store", "data")],
    State("series-store", "data"),
    prevent_initial_call=True,
)
def render_inventory(lead_time, service_level, best_payload, data):
    serie = _serie_desde_store(data)
    if serie is None or best_payload is None or not lead_time:
        raise PreventUpdate
    try:
        spec = get_spec(best_payload["metodo"])
        pol = compute_policy(serie, spec, best_payload["params"], lead_time=int(lead_time),
                             service_level=float(service_level), season_length=12)
    except Exception as exc:
        logger.exception("Fallo el calculo de politica de inventario")
        return dbc.Alert("Error al calcular la politica: {}".format(exc), color="danger")

    body = [
        html.P("Demanda esperada durante el lead time: {:,.0f}".format(pol.demand_lead_time)),
        html.P("Sigma del error acumulado: {:,.0f} ({})".format(
            pol.sigma_lead_time, pol.sigma_method)),
        html.P("Stock de seguridad: {:,.0f}".format(pol.safety_stock)),
        html.P("Punto de reorden: {:,.0f}".format(pol.reorder_point), className="fw-bold"),
    ]
    alertas = [dbc.Alert(w, color="warning", className="small py-2") for w in pol.warnings]
    return dbc.Card([dbc.CardHeader("Politica de inventario"),
                     dbc.CardBody(body + alertas)])


if __name__ == "__main__":
    debug = os.environ.get("FORECASTING_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="127.0.0.1", port=int(os.environ.get("PORT", 8050)))
