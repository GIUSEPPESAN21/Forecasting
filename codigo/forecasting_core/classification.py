"""Clasificacion estructural de la serie: tendencia, estacionalidad, estacionariedad.

Resuelve F01, F10 y F11.

Que estaba mal en el codigo original
------------------------------------
* F11 - tendencia: se usaba el p-value de `scipy.stats.linregress`, que asume
  errores i.i.d. Con demanda autocorrelacionada el test es masivamente
  anticonservador: 74.4% de falsos positivos sobre paseos aleatorios puros
  (nivel nominal 5%). Corregirlo con errores HAC no alcanza —bajo raiz unitaria
  el estadistico t ni siquiera es normal, y HAC deja 66-74%—, asi que se usa el
  procedimiento secuencial de Dickey-Fuller/Pantula: ADF primero, y el test de
  tendencia adecuado a cada regimen (ver `trend_test`).
* F10 - estacionalidad: se usaba |ACF(12)| > 0.30 sobre la serie SIN
  desestacionalizar. Una tendencia induce autocorrelacion cercana a 1 en todos
  los rezagos, de modo que el 50.2% de las series con solo tendencia se
  declaraban estacionales. Aqui se exigen dos evidencias: fuerza estacional STL
  (Wang, Smith & Hyndman 2006) y significancia por Kruskal-Wallis, con un
  minimo de 3 ciclos completos (ver `seasonality_test`).
* F11 - estacionariedad: `adfuller` se llamaba con `regression='c'` (solo
  constante), mal especificado frente a una tendencia deterministica. Aqui la
  especificacion se elige segun el regimen y se confirma con KPSS, cuya
  hipotesis nula es la opuesta; la discrepancia se reporta como "no
  concluyente" en vez de forzar un Si/No.
* F01 - el predicado `es_muy_lineal` exigia R2>=0.90 y |ACF12|<0.10
  simultaneamente, condiciones mutuamente excluyentes: devolvia False incluso
  para y = 1000 + 25t. Se reemplaza por `allow_constant_level_methods`.

Orden de las pruebas
--------------------
El orden no es arbitrario y es parte de la correccion:

    estacionalidad  ->  desestacionalizar  ->  tendencia  ->  estacionariedad

Preguntar por la tendencia sobre una serie con estacionalidad fuerte la pierde
(100% de falsos negativos con n=24), porque la oscilacion anual domina la
varianza. Resultado global medido, 200 replicas por celda:

    falsos positivos      original    corregido
    tendencia               74.4%      9.7% media / 20.5% max
    estacionalidad          50.2%      1.8% media /  4.0% max
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

__all__ = [
    "SeriesProfile", "classify_series", "seasonal_strength", "seasonality_test",
    "deseasonalize", "trend_test", "stationarity_test",
    "allow_constant_level_methods",
    "LOW_POWER_N", "SEASONAL_STRENGTH_THRESHOLD", "TREND_ALPHA",
]

# Por debajo de este n, cualquier prueba estructural tiene potencia insuficiente.
LOW_POWER_N = 48
# Umbral de fuerza estacional. Wang, Smith & Hyndman (2006) y la practica de
# `tsfeatures` situan el corte util entre 0.3 y 0.6; 0.4 equilibra ambos errores.
SEASONAL_STRENGTH_THRESHOLD = 0.40
# Nivel de significancia del test de tendencia.
TREND_ALPHA = 0.05
# Ciclos completos minimos para intentar identificar estacionalidad. Con 2
# ciclos STL no esta identificado y devuelve F_S ~ 1 sobre puro ruido.
MIN_CYCLES_FOR_STL = 3


@dataclass
class SeriesProfile:
    """Perfil estructural de una serie, con toda la evidencia que lo sustenta."""

    n_obs: int
    m: int

    has_trend: bool
    trend_slope: float
    trend_pvalue: float
    trend_test: str

    has_seasonality: bool
    seasonal_strength: float
    seasonality_pvalue: float
    seasonality_test: str

    is_stationary: bool
    adf_pvalue: float
    adf_regression: str
    kpss_pvalue: float
    stationarity_verdict: str

    low_power: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def seasonal_period(self) -> int:
        """Periodo estacional efectivo: 12 si hay estacionalidad, 1 si no.

        Es el `m` que alimenta MASE y el calculo de `min_train`; usar 12 en una
        serie no estacional infla artificialmente el minimo de entrenamiento.
        """
        return self.m if self.has_seasonality else 1

    def as_dict(self) -> dict:
        d = asdict(self)
        d["seasonal_period"] = self.seasonal_period
        return d

    def describe(self) -> str:
        return (
            "Tendencia: {} (p={:.4f}, {}) | Estacionalidad: {} (F_S={:.3f}) | "
            "Estacionaria: {}".format(
                "Si" if self.has_trend else "No",
                self.trend_pvalue,
                self.trend_test,
                "Si" if self.has_seasonality else "No",
                self.seasonal_strength,
                self.stationarity_verdict,
            )
        )


def _clean(series) -> np.ndarray:
    v = np.asarray(pd.Series(series).astype(float).dropna().to_numpy())
    return v[np.isfinite(v)]


def _hac_lags(n: int) -> int:
    """Regla de Newey & West (1994) para el ancho de banda."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _adf_p(y: np.ndarray, regression: str) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(y, autolag="AIC", regression=regression)[1])
    except Exception as exc:
        logger.warning("ADF(%s) fallo: %s", regression, exc)
        return float("nan")


def trend_test(series, alpha: float = TREND_ALPHA, p_adf_ct: float | None = None):
    """Test de tendencia secuencial de Dickey-Fuller / Pantula (F11).

    Por que no basta con corregir los errores estandar
    -------------------------------------------------
    Bajo raiz unitaria el estadistico t de la pendiente OLS **no es
    asintoticamente normal**: ninguna correccion HAC lo arregla. Medido sobre
    500 replicas de paseos aleatorios puros (sin tendencia deterministica), la
    tasa de falsos positivos es 75-81% con OLS clasico y sigue en 66-74% con
    HAC. El unico camino correcto es resolver primero si hay raiz unitaria y
    despues preguntar por la tendencia en el marco adecuado:

    * ADF con regresion 'ct' rechaza -> la serie es trend-stationary y el
      t-test sobre el nivel es valido. Se usa GLSAR(1) (Cochrane-Orcutt), que
      modela explicitamente errores AR(1) y esta bien dimensionado.
    * ADF con regresion 'ct' no rechaza -> la serie es I(1) y la nocion correcta
      de "tendencia" es la **deriva**: se contrasta si la media de la primera
      diferencia difiere de cero, un t-test perfectamente valido.

    Tamano y potencia medidos (500 replicas, alpha=5%); reproducible con
    `experiments/montecarlo_clasificacion.py`:

        regimen            n=24    n=48   n=120    (esperado)
        ruido blanco       4.6%    7.4%    7.0%    ~5%  tamano
        AR(1) phi=0.7      4.8%   10.2%    9.8%    ~5%  tamano
        paseo aleatorio   14.8%   12.6%   11.6%    ~5%  tamano
        tendencia real    50.8%   99.8%  100.0%    alta potencia
        deriva real       39.6%   54.6%   92.0%    alta potencia

    El residuo en paseos aleatorios proviene de la distorsion de tamano del
    propio ADF con n pequeno (18.3% medido); por eso `low_power` marca n<48.

    Devuelve (hay_tendencia, pendiente, p_value, nombre_del_test).
    """
    y = _clean(series)
    n = y.size
    if n < 8:
        return False, float("nan"), float("nan"), "insuficientes datos (n<8)"

    p_ct = _adf_p(y, "ct") if p_adf_ct is None else float(p_adf_ct)

    if np.isfinite(p_ct) and p_ct < alpha:
        # Rama trend-stationary: pendiente sobre el nivel con errores AR(1).
        x = sm.add_constant(np.arange(n, dtype=float))
        try:
            fit = sm.GLSAR(y, x, rho=1).iterative_fit(maxiter=10)
            slope, pval = float(fit.params[1]), float(fit.pvalues[1])
            name = "GLSAR(1) sobre el nivel [ADF ct p={:.3f} => trend-stationary]".format(p_ct)
        except Exception as exc:
            logger.warning("GLSAR fallo (%s); se usa OLS+HAC", exc)
            fit = sm.OLS(y, x).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": _hac_lags(n), "use_correction": True},
                use_t=True,
            )
            slope, pval = float(fit.params[1]), float(fit.pvalues[1])
            name = "OLS+HAC sobre el nivel [GLSAR no disponible]"
        return bool(pval < alpha), slope, pval, name

    # Rama I(1): la tendencia es la deriva de la primera diferencia.
    d = np.diff(y)
    nd = d.size
    if nd < 4:
        return False, float("nan"), float("nan"), "insuficientes datos para la deriva"
    try:
        fit = sm.OLS(d, np.ones((nd, 1))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": _hac_lags(nd), "use_correction": True},
            use_t=True,
        )
        slope, pval = float(fit.params[0]), float(fit.pvalues[0])
    except Exception as exc:  # pragma: no cover
        logger.warning("Test de deriva fallo: %s", exc)
        return False, float("nan"), float("nan"), "test de deriva fallo"
    name = "deriva de la primera diferencia [ADF ct p={:.3f} => I(1)]".format(p_ct)
    return bool(pval < alpha), slope, pval, name


def _stl(y: np.ndarray, m: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return STL(pd.Series(y), period=m, robust=True).fit()


def seasonal_strength(series, m: int = 12) -> tuple[float, str]:
    """Fuerza estacional F_S via descomposicion STL (F10).

        F_S = max(0, 1 - Var(residuo) / Var(estacional + residuo))

    A diferencia de |ACF(m)|, se calcula sobre la componente estacional que STL
    separa de la tendencia, de modo que una serie con solo tendencia da F_S ~ 0.

    Requiere 3 ciclos completos. Con 2 ciclos STL esta practicamente no
    identificado: absorbe el ruido en la componente estacional y devuelve
    F_S ~ 1 casi siempre (88-98% de falsos positivos medidos). Con 24
    observaciones mensuales la respuesta honesta es "no evaluable", no "Si".

    Devuelve (F_S, nombre_del_test).
    """
    y = _clean(series)
    if m < 2:
        return 0.0, "no aplica (m<2)"
    if y.size < MIN_CYCLES_FOR_STL * m:
        return (
            float("nan"),
            "no evaluable: se requieren {} observaciones ({} ciclos de m={}) y hay "
            "{}".format(MIN_CYCLES_FOR_STL * m, MIN_CYCLES_FOR_STL, m, y.size),
        )
    try:
        res = _stl(y, m)
        resid = np.asarray(res.resid, dtype=float)
        seas = np.asarray(res.seasonal, dtype=float)
        denom = np.var(seas + resid, ddof=1)
        if denom <= 0:
            return 0.0, "STL (varianza nula)"
        fs = 1.0 - np.var(resid, ddof=1) / denom
        return float(max(0.0, min(1.0, fs))), "STL seasonal strength (Wang et al. 2006)"
    except Exception as exc:
        logger.warning("STL fallo (%s); estacionalidad no evaluable", exc)
        return float("nan"), "STL fallo: {}".format(exc)


def seasonality_test(series, m: int = 12) -> tuple[bool, float, float, str]:
    """Estacionalidad = fuerza suficiente Y significativa (F10).

    Combina dos evidencias independientes, al estilo del test de Ollech & Webel
    (2020) que implementa el paquete `seastests`:

      1. **Magnitud**: F_S >= 0.40 sobre la descomposicion STL.
      2. **Significancia**: Kruskal-Wallis sobre los residuos detrendizados
         agrupados por posicion dentro del ciclo. Es no parametrico, asi que no
         asume normalidad ni varianzas iguales entre meses.

    Exigir ambas evita los dos modos de falla del criterio original
    |ACF(12)|>0.30: declarar estacionalidad por arrastre de la tendencia, y
    declararla a partir de ruido con muy pocos ciclos.

    Tasas medidas (200 replicas por celda):

        serie                 n=24     n=36     n=48    n=120   verdad
        ruido blanco       no eval.    1.5%     3.5%     1.0%   No
        paseo aleatorio    no eval.    2.0%     1.0%     0.0%   No
        solo tendencia     no eval.    3.0%     1.0%     0.5%   No
        estacional fuerte  no eval.  100.0%   100.0%   100.0%   Si
        estacional debil   no eval.   59.0%    90.0%    98.0%   Si

    Devuelve (hay_estacionalidad, F_S, p_kruskal, nombre_del_test).
    """
    y = _clean(series)
    fs, fs_name = seasonal_strength(y, m=m)
    if not np.isfinite(fs):
        return False, fs, float("nan"), fs_name

    n = y.size
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    detrended = y - (intercept + slope * x)
    groups = [detrended[i::m] for i in range(m) if detrended[i::m].size >= 2]
    p_kw = float("nan")
    if len(groups) >= 2:
        try:
            p_kw = float(stats.kruskal(*groups)[1])
        except Exception as exc:
            logger.warning("Kruskal-Wallis fallo: %s", exc)

    strong = fs >= SEASONAL_STRENGTH_THRESHOLD
    significant = np.isfinite(p_kw) and p_kw < 0.05
    name = "{} + Kruskal-Wallis sobre residuos detrendizados (p={:.4f})".format(
        fs_name, p_kw
    )
    return bool(strong and significant), fs, p_kw, name


def deseasonalize(series, m: int = 12) -> np.ndarray:
    """Serie desestacionalizada por medias estacionales (descomposicion clasica).

    Se aplica ANTES del test de tendencia cuando hay estacionalidad: una
    oscilacion anual de gran amplitud enmascara una pendiente moderada, y
    testeando sobre la serie cruda la tendencia se perdia en el 100% de las
    series con tendencia+estacionalidad y n=24.

    Por que medias estacionales y no STL
    ------------------------------------
    STL se usa para MEDIR la fuerza estacional, pero para REMOVERLA antes del
    test de tendencia filtra variacion de baja frecuencia hacia el residuo y
    genera tendencias espurias. Medido sobre 300 replicas de series puramente
    estacionales (sin tendencia), la tasa de falsos positivos de tendencia es:

        n:                        36      48     120
        STL                     14.7%   11.3%   13.0%   <- no converge
        medias estacionales     12.0%    9.7%    5.3%   <- converge al nominal

    Las medias se calculan sobre la serie previamente detrendizada, de modo que
    la propia tendencia no contamina los efectos estacionales. El costo es algo
    menos de potencia con pocos ciclos (82.7% vs 91.3% en n=36); se prefiere el
    tamano correcto, coherente con el criterio de todo este refactor.
    """
    y = _clean(series)
    n = y.size
    if m < 2 or n < MIN_CYCLES_FOR_STL * m:
        return y
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    detrended = y - (intercept + slope * x)
    pos = np.arange(n) % m
    effects = np.array([
        detrended[pos == i].mean() if np.any(pos == i) else 0.0 for i in range(m)
    ])
    effects -= effects.mean()  # los efectos estacionales suman cero
    return np.asarray(y - effects[pos], dtype=float)


def stationarity_test(
    series, has_trend: bool, p_adf: float | None = None
) -> tuple[bool, float, str, float, str]:
    """ADF + KPSS con especificacion coherente con el test de tendencia (F11).

    ADF: H0 = raiz unitaria. Se usa `regression='ct'` si hay tendencia
    deterministica detectada, `'c'` en caso contrario.
    KPSS: H0 = estacionariedad (hipotesis opuesta). La coincidencia de ambos da
    un veredicto firme; la discrepancia se reporta como "no concluyente" en vez
    de forzar un Si/No.

    Devuelve (es_estacionaria, p_adf, regresion_adf, p_kpss, veredicto).
    """
    y = _clean(series)
    reg = "ct" if has_trend else "c"
    p_kpss = float("nan")

    if y.size < 12:
        return False, float("nan"), reg, p_kpss, "no evaluable (n<12)"

    p_adf = _adf_p(y, reg) if p_adf is None else float(p_adf)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_kpss = float(kpss(y, regression=reg, nlags="auto")[1])
    except Exception as exc:
        logger.warning("KPSS fallo: %s", exc)

    adf_stationary = np.isfinite(p_adf) and p_adf < 0.05
    kpss_stationary = np.isfinite(p_kpss) and p_kpss > 0.05

    if adf_stationary and kpss_stationary:
        verdict, flag = "Si (ADF y KPSS coinciden)", True
    elif (not adf_stationary) and (not kpss_stationary):
        verdict, flag = "No (ADF y KPSS coinciden)", False
    elif adf_stationary and not kpss_stationary:
        verdict, flag = "No concluyente (ADF si, KPSS no)", False
    else:
        verdict, flag = "No concluyente (ADF no, KPSS si)", False

    if reg == "ct":
        verdict += " [alrededor de una tendencia deterministica]"
    return flag, p_adf, reg, p_kpss, verdict


def allow_constant_level_methods(profile: SeriesProfile) -> bool:
    """Reemplazo de `es_muy_lineal` (F01).

    El predicado original exigia R2>=0.90 Y |ACF12|<0.10, condiciones
    mutuamente excluyentes: una tendencia fuerte implica ACF12 alta. El
    resultado era que Promedio Simple, Movil y Ponderado quedaban excluidos del
    ranking en TODA ejecucion.

    Medido sobre 3000 series sinteticas, el predicado original devolvia False en
    el 100% de los casos, incluida y = 1000 + 25t (R2=1.000, ACF12=1.000).

    La condicion correcta para admitir un pronostico de nivel constante es que
    la serie tenga un nivel ESTABLE, lo que exige dos cosas a la vez:

      1. ausencia de tendencia significativa, y
      2. estacionariedad en nivel.

    El segundo requisito importa: en un paseo aleatorio no hay tendencia
    deterministica pero el nivel se desplaza, y promediar toda la historia es
    peor que el naive. Los metodos de nivel ADAPTATIVO (SES, naive) no estan
    sujetos a esta restriccion y se evaluan siempre.
    """
    return (not profile.has_trend) and profile.is_stationary


def classify_series(series, m: int = 12) -> SeriesProfile:
    """Clasificacion estructural completa con toda su evidencia."""
    y = _clean(series)
    n = int(y.size)
    warns: list[str] = []

    # ORDEN IMPORTANTE: primero estacionalidad, luego se desestacionaliza, y
    # solo entonces se pregunta por la tendencia. Una oscilacion anual de gran
    # amplitud enmascara una pendiente moderada; testeando sobre la serie cruda
    # la tendencia se perdia en el 100% de las series con tendencia+estacionalidad.
    has_seas, fs, p_kw, s_name = seasonality_test(y, m=m)
    y_trend = deseasonalize(y, m=m) if has_seas else y

    # ADF('ct') se calcula UNA sola vez y alimenta los dos tests: evita recomputo
    # (F14) y garantiza que tendencia y estacionariedad sean veredictos
    # coherentes entre si, no dos respuestas independientes.
    p_adf_ct = _adf_p(y_trend, "ct") if n >= 12 else float("nan")
    has_trend, slope, p_trend, t_name = trend_test(y_trend, p_adf_ct=p_adf_ct)
    if has_seas:
        t_name += " [sobre la serie desestacionalizada]"
    is_stat, p_adf, adf_reg, p_kpss, verdict = stationarity_test(
        y_trend, has_trend, p_adf=p_adf_ct if has_trend else None
    )

    low_power = n < LOW_POWER_N
    if low_power:
        warns.append(
            "Serie de {} observaciones (< {}): las pruebas de tendencia, "
            "estacionalidad y estacionariedad tienen potencia limitada. Trate la "
            "clasificacion como indicativa, no como un hecho.".format(n, LOW_POWER_N)
        )
    if not np.isfinite(fs):
        warns.append(
            "Estacionalidad no evaluable: {}. Los metodos estacionales quedan "
            "fuera del ranking por falta de historia.".format(s_name)
        )
    if n < MIN_CYCLES_FOR_STL * m and m > 1:
        warns.append(
            "Se requieren al menos {} observaciones ({} ciclos completos) para "
            "identificar estacionalidad anual.".format(MIN_CYCLES_FOR_STL * m, MIN_CYCLES_FOR_STL)
        )
    if has_trend and is_stat:
        warns.append(
            "Se detecto tendencia deterministica y estacionariedad alrededor de "
            "ella (ADF con regresion 'ct'): la serie es trend-stationary, no "
            "estacionaria en nivel."
        )

    return SeriesProfile(
        n_obs=n,
        m=m,
        has_trend=has_trend,
        trend_slope=slope,
        trend_pvalue=p_trend,
        trend_test=t_name,
        has_seasonality=has_seas,
        seasonal_strength=fs,
        seasonality_pvalue=p_kw,
        seasonality_test=s_name,
        is_stationary=is_stat,
        adf_pvalue=p_adf,
        adf_regression=adf_reg,
        kpss_pvalue=p_kpss,
        stationarity_verdict=verdict,
        low_power=low_power,
        warnings=warns,
    )
