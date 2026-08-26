"""Envoltorios `ModelSpec` para Prophet y LightGBM (Fase 11, F27).

Por que instancias REALES de `ModelSpec` y no un duck-type
-------------------------------------------------------------
`forecasting_core.models.ModelSpec` es un `dataclass` congelado cuyo unico
requisito de comportamiento es un callable `_forecast(y, params, h, m)`. Eso
significa que la forma mas simple de que un adaptador externo tenga
"la MISMA forma que ModelSpec" es que literalmente SEA un `ModelSpec` -no
hace falta imitarlo. `PROPHET_SPEC`/`LIGHTGBM_SPEC` de aqui son instancias
reales, construidas con una funcion `_forecast` que despacha (de forma
perezosa) al adaptador correspondiente.

Por que esto NO necesito el "hook aditivo" que anticipaba el prompt de la
Fase 11 en `optimize.py`
-------------------------------------------------------------------------
El prompt preveia anadir un hook a `honest_outer_estimate`/`optimize.py` por
si esa funcion exigiera especificamente un `ModelSpec` sacado de
`MODEL_REGISTRY` y no aceptara instancias externas. En la practica:

* `forecasting_core.validation.walk_forward` y `backtest_one_step` aceptan
  cualquier lista de `ModelSpec` directamente -MODEL_REGISTRY no interviene
  en absoluto en esa capa. `PROPHET_SPEC`/`LIGHTGBM_SPEC` se le pueden pasar
  sin ningun cambio al nucleo (ver el test
  `test_specs_plug_into_walk_forward` en
  `codigo/tests/test_external_baselines.py`).
* `honest_outer_estimate`/`run_pipeline` SI estan acoplados a
  `MODEL_REGISTRY` (via `eligible_specs`), porque su trabajo es justamente
  ELEGIR entre los modelos del registro con el filtro estructural (F22). Pero
  `codigo/experimentos/comparativa_externa.py` no necesita que el nucleo
  "elija" entre la Herramienta y los externos: evalua Prophet/LightGBM
  directamente sobre el mismo bloque EXTERNO que `honest_outer_estimate` ya
  reservo para la Herramienta -exactamente el patron que
  `experimentos/vs_incumbente.py` ya usa para el metodo incumbente (que
  tampoco pasa por `MODEL_REGISTRY`/`eligible_specs`).

Por eso `codigo/forecasting_core/optimize.py` queda intacto en esta fase: no
hizo falta el hook. Esta decision queda documentada aqui y en el CHANGELOG
bajo F27 para que quede trazable, tal como pedia el prompt maestro incluso
para la rama donde el hook resulta innecesario.
"""
from __future__ import annotations

import numpy as np

from forecasting_core.models import ModelSpec

from .lightgbm_adapter import LGBM_MIN_OBS, fit_predict_lgbm
from .prophet_adapter import PROPHET_MIN_OBS, fit_predict_prophet

__all__ = ["PROPHET_SPEC", "LIGHTGBM_SPEC", "external_specs"]


def _forecast_prophet(y, p, h, m):
    return fit_predict_prophet(y, int(h), freq="MS", clip_non_negative=True)


def _forecast_lgbm(y, p, h, m):
    return fit_predict_lgbm(y, int(h), freq="MS", clip_non_negative=True)


PROPHET_SPEC = ModelSpec(
    key="ext_prophet",
    label="Prophet (backend Stan)",
    family="external_baseline",
    seasonal=True,
    constant_level=False,
    _forecast=_forecast_prophet,
    _min_obs=lambda m: PROPHET_MIN_OBS,
    analytic_interval=False,
    note="Comparador externo aislado (F27). yearly_seasonality solo se activa "
         "con >=24 observaciones (2 ciclos anuales completos); ver F26 y "
         "prophet_adapter.py. Sin tuning de changepoint_prior_scale ni de "
         "otros hiperparametros de Prophet (linea base, no tesis de tuning).",
)

LIGHTGBM_SPEC = ModelSpec(
    key="ext_lightgbm",
    label="LightGBM (mlforecast, modelo global)",
    family="external_baseline",
    seasonal=True,
    constant_level=False,
    _forecast=_forecast_lgbm,
    _min_obs=lambda m: LGBM_MIN_OBS,
    analytic_interval=False,
    note="Comparador externo aislado (F27). Features de lags/rolling estandar "
         "de mlforecast (familia Nixtla, consistente con statsforecast ya "
         "usado en el nucleo). Sin busqueda de hiperparametros de LightGBM.",
)


def external_specs() -> dict[str, ModelSpec]:
    """Adaptadores disponibles SEGUN lo instalado en este entorno.

    No lanza excepcion si falta un paquete: simplemente lo omite del dict.
    `comparativa_externa.py` y la pestana de `app.py` usan esto para
    degradarse con gracia (F27 seccion 3 del prompt de la Fase 11: "no dejes
    la fase completa bloqueada por un solo paquete").
    """
    out: dict[str, ModelSpec] = {}
    try:
        import prophet  # noqa: F401

        out["ext_prophet"] = PROPHET_SPEC
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401
        import mlforecast  # noqa: F401

        out["ext_lightgbm"] = LIGHTGBM_SPEC
    except ImportError:
        pass
    return out
