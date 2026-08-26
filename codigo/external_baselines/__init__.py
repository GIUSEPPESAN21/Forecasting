"""Comparadores externos (Fase 11 / F27) — Prophet y LightGBM.

Paquete HERMANO de `forecasting_core`, deliberadamente aislado de él: nada
aquí se importa desde `forecasting_core`, `app.py` ni `batch_cli.py` a nivel
de módulo. Los dos backends pesados (`prophet`/`cmdstanpy`, `mlforecast`/
`lightgbm`) se importan de forma perezosa, dentro de cada función, para que
`import forecasting_core` y `python codigo/app.py` sigan funcionando sin
estos paquetes instalados (ver `requirements-external.txt`).

Ver `codigo/experimentos/decision_prophet.md` para la decisión original de
retirar Prophet del manuscrito (F08) y la razón por la que esta fase lo
reincorpora como comparador aislado bajo instrucción explícita de los
tutores del proyecto (F26-F27).
"""
from __future__ import annotations

__all__ = ["PROPHET_AVAILABLE", "LIGHTGBM_AVAILABLE"]


def _probe(module: str) -> bool:
    """Comprueba si `module` es importable SIN importarlo.

    `importlib.util.find_spec` solo consulta a los finders del sistema de
    import; no ejecuta el modulo. Es deliberado: `app.py` llama a este
    modulo a nivel de import para decidir si mostrar el Modulo 5
    (Comparacion externa) habilitado o no, y prophet/mlforecast son pesados
    de importar de verdad (cmdstanpy, lightgbm, sklearn) — cargarlos solo
    para una comprobacion de disponibilidad violaria el presupuesto de RAM
    de la sesion interactiva (ver prompt maestro, Fase 11 §1).
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


PROPHET_AVAILABLE = _probe("prophet")
LIGHTGBM_AVAILABLE = _probe("mlforecast") and _probe("lightgbm")
