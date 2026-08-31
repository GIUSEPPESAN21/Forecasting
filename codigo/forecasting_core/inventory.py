"""Del pronostico a la decision de inventario (F20, M-08).

La tesis prometia "reducir la recurrencia de sobrestocks y faltantes", pero la
salida de la herramienta era un Excel con `year, month, forecast`. Entre eso y
una decision de inventario faltaban la distribucion del error, el nivel de
servicio, el lead time, el punto de reorden y el stock de seguridad.

Este modulo cierra ese tramo. La formulacion es la clasica de Silver, Pyke &
Peterson (1998) y Nahmias & Olsen (2015):

    sigma_L = desviacion del error de pronostico ACUMULADO sobre el lead time
    SS      = z(nivel de servicio) * sigma_L
    ROP     = demanda esperada durante el lead time + SS

Detalle que importa y suele omitirse: sigma_L **no** es `sigma_1 * sqrt(L)`
salvo que los errores de periodos sucesivos sean independientes, cosa que casi
nunca ocurre en pronosticos (un modelo sesgado se equivoca en la misma
direccion varios meses seguidos). Aqui sigma_L se estima directamente de los
errores acumulados observados en el backtest, y el metodo usado queda
declarado en el resultado.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from .models import ModelSpec
from .validation import multi_horizon_errors

logger = logging.getLogger(__name__)

__all__ = ["InventoryPolicy", "safety_stock", "compute_policy", "SERVICE_LEVELS"]

SERVICE_LEVELS = (0.90, 0.95, 0.975, 0.99)


@dataclass
class InventoryPolicy:
    lead_time: int
    service_level: float
    z: float
    demand_lead_time: float
    sigma_lead_time: float
    safety_stock: float
    reorder_point: float
    sigma_method: str
    bias_lead_time: float
    warnings: list[str]

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return (
            "Lead time {} meses | nivel de servicio {:.1%} (z={:.2f})\n"
            "  Demanda esperada en el lead time : {:,.0f}\n"
            "  Sigma del error acumulado        : {:,.0f}  [{}]\n"
            "  Stock de seguridad               : {:,.0f}\n"
            "  Punto de reorden                 : {:,.0f}".format(
                self.lead_time, self.service_level, self.z,
                self.demand_lead_time, self.sigma_lead_time, self.sigma_method,
                self.safety_stock, self.reorder_point,
            )
        )


def safety_stock(sigma_lead_time: float, service_level: float = 0.95) -> tuple[float, float]:
    """SS = z * sigma_L. Devuelve (SS, z).

    F38: de las dos piezas de la formula, solo `sigma_L` es empirica; `z` es
    el cuantil de la normal ESTANDAR que corresponde al nivel de servicio
    pedido, `z = Phi^-1(nivel_de_servicio)` (p.ej. z=1.645 para 95%), tal
    cual la formulacion clasica de Silver, Pyke & Peterson (1998). Lo que
    distingue a este modulo de la aproximacion de libro de texto NO es
    reemplazar ese cuantil por uno empirico -sigue siendo normal-, sino
    estimar `sigma_L` (el error acumulado sobre el lead time) directamente
    de los residuos del walk-forward en `compute_policy`, en vez de asumir
    `sigma_L = sigma_1 * sqrt(L)` (independencia entre errores sucesivos).
    """
    if not 0.5 < service_level < 1.0:
        raise ValueError("El nivel de servicio debe estar entre 0.5 y 1.0")
    z = float(stats.norm.ppf(service_level))
    return float(z * sigma_lead_time), z


def compute_policy(
    series,
    spec: ModelSpec,
    params: dict | None,
    *,
    lead_time: int = 3,
    service_level: float = 0.95,
    season_length: int = 12,
    n_origins: int = 10,
    clip_non_negative: bool = True,
) -> InventoryPolicy:
    """Politica (s, Q) a partir del error de pronostico medido en backtest."""
    s = pd.Series(series).astype(float).dropna()
    y = s.to_numpy(dtype=float)
    L = int(lead_time)
    if L < 1:
        raise ValueError("El lead time debe ser de al menos 1 periodo")

    warns: list[str] = []

    forecast = spec.forecast(
        y, params=params, h=L, m=season_length, clip_non_negative=clip_non_negative
    )
    demand_L = float(np.sum(forecast))

    errs = multi_horizon_errors(
        y, spec, params, season_length, L,
        n_origins=n_origins, clip_non_negative=clip_non_negative,
    )
    # Error ACUMULADO sobre el lead time en cada origen: captura la correlacion
    # temporal de los errores, que la formula sigma_1*sqrt(L) ignora. Solo se
    # usan los origenes con los L horizontes completos.
    valid_rows = np.isfinite(errs).all(axis=1)
    cum = errs[valid_rows].sum(axis=1) if valid_rows.any() else np.array([])

    if cum.size >= 3:
        sigma_L = float(np.std(cum, ddof=1))
        bias_L = float(np.mean(cum))
        method = "desviacion de los errores acumulados sobre {} origenes".format(cum.size)
    else:
        one_step = errs[:, 0]
        one_step = one_step[np.isfinite(one_step)]
        if one_step.size >= 2:
            sigma_1 = float(np.std(one_step, ddof=1))
            bias_1 = float(np.mean(one_step))
        else:
            sigma_1 = float(np.std(np.diff(y), ddof=1))
            bias_1 = 0.0
            warns.append(
                "No hubo errores de backtest utilizables; sigma se aproximo con la "
                "desviacion de la primera diferencia de la serie."
            )
        sigma_L = sigma_1 * np.sqrt(L)
        bias_L = bias_1 * L
        method = (
            "aproximacion sigma_1*sqrt(L): solo {} origenes completos, "
            "insuficientes para medir el error acumulado. ASUME errores "
            "independientes entre periodos, lo que SUBESTIMA sigma_L si el "
            "modelo esta sesgado.".format(int(valid_rows.sum()))
        )
        warns.append(
            "Estimacion de sigma_L por aproximacion: trate el stock de seguridad "
            "resultante como un piso, no como un valor calibrado."
        )

    ss, z = safety_stock(sigma_L, service_level)
    rop = demand_L + ss

    if abs(bias_L) > 0.5 * sigma_L and sigma_L > 0:
        warns.append(
            "El sesgo acumulado sobre el lead time ({:+,.0f}) supera medio sigma "
            "({:,.0f}): el modelo se equivoca sistematicamente en una direccion. "
            "Corrija el sesgo antes de dimensionar el stock; el stock de seguridad "
            "protege contra variabilidad, no contra sesgo.".format(bias_L, sigma_L)
        )

    return InventoryPolicy(
        lead_time=L, service_level=float(service_level), z=z,
        demand_lead_time=demand_L, sigma_lead_time=sigma_L,
        safety_stock=ss, reorder_point=rop, sigma_method=method,
        bias_lead_time=bias_L, warnings=warns,
    )
