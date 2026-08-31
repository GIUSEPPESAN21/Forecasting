"""F31 - la banda de prediccion no puede angostarse con el horizonte.

Evidencia real que motivo el hallazgo (resultados/caso_ilustrativo_pronostico.csv,
version anterior al fix): sigma por horizonte = 360, 347, 192, 204, 99, 31, 57,
65, 24, 1140, 1195, 1248 - se estrecha casi a cero en los meses 6-9 y explota en
los meses 10-12. Con menos de MIN_ORIGINS_FOR_EMPIRICAL origenes por horizonte,
sigma se estima con muestras muy pequenas y el ruido de muestreo puede producir
justamente ese patron no monotono, que no tiene sentido fisico (la incertidumbre
sobre el futuro no puede DISMINUIR al mirar mas lejos).
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting_core.intervals import prediction_interval
from forecasting_core.models import MODEL_REGISTRY
from conftest import make_series


def _columna_con_std(objetivo: float, n: int, seed: int) -> np.ndarray:
    """n valores centrados en cero cuya desviacion estandar (ddof=1) es `objetivo`."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    base -= base.mean()
    escala = objetivo / np.std(base, ddof=1)
    return base * escala


def test_sigma_no_decreciente_en_horizonte(monkeypatch):
    """Caso sintetico que reproduce la banda no monotona reportada en la evidencia."""
    horizon = 12
    n_origins = 10
    # sigma "crudo" no monotono: se angosta en 4-8 y explota en 9-11 (0-indexado).
    sigmas_crudos = [360, 347, 192, 204, 99, 31, 57, 65, 24, 1140, 1195, 1248]
    errs = np.column_stack([
        _columna_con_std(s, n_origins, seed=100 + h) for h, s in enumerate(sigmas_crudos)
    ])

    monkeypatch.setattr(
        "forecasting_core.intervals.multi_horizon_errors",
        lambda *a, **k: errs,
    )

    spec = MODEL_REGISTRY["naive"]
    s = make_series("tendencia_estacional", 48)
    # clip_non_negative=False: se aisla el invariante de monotonicidad del
    # ancho de banda del recorte por no-negatividad de la demanda, que es una
    # restriccion de dominio legitima y puede angostar el limite inferior de
    # forma asimetrica cerca de cero sin que eso sea un defecto de F31.
    interval = prediction_interval(s, spec, None, horizon=horizon, n_origins=n_origins,
                                    clip_non_negative=False)

    sigma = interval.sigma
    for h in range(1, horizon):
        assert sigma[h] >= sigma[h - 1] - 1e-9, (
            "sigma no monotono en h={}: {} < {}".format(h, sigma[h], sigma[h - 1])
        )

    ancho = interval.upper - interval.lower
    for h in range(1, horizon):
        assert ancho[h] >= ancho[h - 1] - 1e-9, (
            "ancho de banda no monotono en h={}: {} < {}".format(h, ancho[h], ancho[h - 1])
        )


@pytest.mark.parametrize("kind", ["tendencia_estacional", "plana", "paseo_aleatorio"])
def test_banda_monotona_en_series_reales_sin_mock(kind):
    """Sin forzar ningun escenario sintetico, el invariante debe sostenerse siempre."""
    spec = MODEL_REGISTRY["naive"]
    s = make_series(kind, 48)
    interval = prediction_interval(s, spec, None, horizon=12, clip_non_negative=False)
    sigma = interval.sigma
    assert np.all(np.diff(sigma) >= -1e-9)
    ancho = interval.upper - interval.lower
    assert np.all(np.diff(ancho) >= -1e-9)
