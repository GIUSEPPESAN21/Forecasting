"""forecasting_core - nucleo de pronostico, sin dependencias de interfaz.

Este paquete NO importa dash ni plotly: es ejecutable y testeable desde un
script, un notebook o CI. La aplicacion (`app.py`) es una capa delgada encima.

Mapa rapido
-----------
    data            carga y validacion (F15)
    classification  tendencia / estacionalidad / estacionariedad (F01, F10, F11)
    models          registro explicito de modelos (F04)
    metrics         MASE primaria, MAPE seguro ante ceros, ME (F06, F12)
    validation      walk-forward de una sola pasada, con paridad (F02, F03, F14)
    optimize        tuning anidado y ganador sin sesgo de seleccion (F05, F13)
    intervals       intervalos de prediccion empiricos y analiticos (F20)
    inventory       stock de seguridad y punto de reorden (F20, M-08)
    batch           procesamiento multi-SKU con memoria acotada (F19)
"""
from __future__ import annotations

__version__ = "2.0.0"

from .classification import SeriesProfile, classify_series  # noqa: F401
from .data import LoadResult, load_panel, load_series, load_series_from_excel  # noqa: F401
from .metrics import compute_metrics, mad, mape, mase, me, mse  # noqa: F401
from .models import MODEL_REGISTRY, ModelSpec, eligible_specs, get_spec  # noqa: F401
from .optimize import PipelineResult, honest_outer_estimate, run_pipeline  # noqa: F401
from .validation import walk_forward  # noqa: F401

__all__ = [
    "__version__",
    "SeriesProfile", "classify_series",
    "LoadResult", "load_series", "load_series_from_excel", "load_panel",
    "compute_metrics", "mase", "mape", "mad", "mse", "me",
    "MODEL_REGISTRY", "ModelSpec", "get_spec", "eligible_specs",
    "walk_forward",
    "run_pipeline", "honest_outer_estimate", "PipelineResult",
]
