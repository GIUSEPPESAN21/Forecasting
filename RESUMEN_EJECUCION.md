# Resumen de ejecución del refactor

Este documento recorre los 25 hallazgos del registro maestro (fusión de las
dos auditorías independientes) y marca el estado final de cada uno. Formato
de estado: `Corregido y verificado` / `Corregido, pendiente de dato externo`
/ `Decisión de diseño aplicada (detallar cuál)`.

## Registro de hallazgos

| ID | Título | Estado | Evidencia |
|---|---|---|---|
| F01 | `es_muy_lineal()` insatisfacible | **Corregido y verificado** | `codigo/forecasting_core/classification.py::allow_constant_level_methods`; `codigo/tests/test_classification.py`; Monte Carlo en `codigo/experimentos/montecarlo_clasificacion.py` |
| F02 | Comparación no homogénea (fallback silencioso HW→Holt, SARIMA→ARIMA) | **Corregido y verificado** | `codigo/forecasting_core/validation.py::walk_forward` (exclusión explícita, nunca promedio sobre menos puntos); `codigo/tests/test_walk_forward_parity.py` |
| F03 | Fuga in-sample en el "MAPE histórico ajustado" | **Corregido y verificado** | Eliminado por construcción: `ModelSpec.forecast()` única ruta backtest/publicado; `codigo/tests/test_no_leakage.py` (perturbación programática del futuro) |
| F04 | `_make_predictor` despacha por subcadena, cae a `train[-1]` | **Corregido y verificado** | `codigo/forecasting_core/models.py::get_spec` (KeyError ante clave desconocida); `codigo/tests/test_model_registry.py` |
| F05 | Sesgo de selección: hiperparámetros elegidos sobre el mismo bloque reportado | **Corregido y verificado** | `codigo/forecasting_core/optimize.py` (partición tune/eval disjunta); `codigo/tests/test_hyperopt_no_bias.py` |
| F06 | Error Medio (ME) declarado, nunca implementado | **Corregido y verificado** | `codigo/forecasting_core/metrics.py::me`; `codigo/tests/test_metrics.py` |
| F07 | `template.tex` no compila, texto de plantilla, declaraciones vacías | **Corregido y verificado*** | `manuscritos/articulo_mdpi/template.tex` reescrito; ver nota de compilación abajo |
| F08 | Sección "Comparison with Prophet" sin protocolo, datos ni código | **Decisión de diseño aplicada: retirada y reemplazada** | `codigo/experimentos/decision_prophet.md`; manuscrito Sección 3.6 (comparación contra AutoARIMA/ETS/Theta) |
| F09 | Afirmación central sin línea base (mejora vs. incumbente) | **Corregido, pendiente de dato externo** | `codigo/experimentos/vs_incumbente.py` — corre sobre dataset sintético; acepta `--input` con el Excel real de la empresa de referencia cuando esté disponible |
| F10 | `\|ACF(12)\|>0.30` sin desestacionalizar (falsos positivos por tendencia) | **Corregido y verificado** | `codigo/forecasting_core/classification.py::seasonality_test` (STL + Kruskal-Wallis); Monte Carlo: 50.2%→1.4% medio |
| F11 | Test de tendencia no robusto a autocorrelación; ADF mal especificado | **Corregido y verificado** | `codigo/forecasting_core/classification.py::trend_test` (procedimiento secuencial ADF→GLSAR/deriva); Monte Carlo: 74.4%→8.9% medio |
| F12 | MAPE explota con demanda cero; sin naive/seasonal_naive; sin MASE | **Corregido y verificado** | `codigo/forecasting_core/metrics.py` (MASE primaria, MAPE excluye ceros); `MODEL_REGISTRY` incluye `naive`/`seasonal_naive` obligatorios |
| F13 | Límites de grid search inconsistentes; Holt/ARIMA/SARIMA sin tope real | **Corregido y verificado** | Parrillas coarse-to-fine acotadas (≤27 combos); ARIMA/SARIMA sin barrido (AICc vía `AutoARIMA`); `codigo/tests/test_model_registry.py::test_las_parrillas_estan_acotadas` |
| F14 | Cómputo redundante: walk-forward ejecutado 3-4 veces por sesión | **Corregido y verificado** | `codigo/forecasting_core/validation.py::walk_forward` (una sola pasada, agregado+detalle); `codigo/tests/test_walk_forward_parity.py::test_walk_forward_devuelve_agregado_y_detalle_en_una_sola_pasada` |
| F15 | Carga de datos frágil (meses, separadores, duplicados, huecos) | **Corregido y verificado** | `codigo/forecasting_core/data.py`; `codigo/tests/test_data_loading.py` (23 pruebas, cubre las 7 variantes) |
| F16 | `except Exception` sin registro; `warnings.filterwarnings("ignore")` global | **Corregido y verificado** | Sin filtro global en ningún módulo de `codigo/forecasting_core/`; logging configurable (`FORECASTING_LOG_LEVEL`) en `codigo/app.py`/`batch_cli.py` |
| F17 | Limitación de demanda por proyectos de obra no declarada específicamente | **Corregido y verificado** | Manuscrito, Sección 4 (Discussion): mención específica a cartera adjudicada/licencias de construcción |
| F18 | Higiene de repositorio (debug=True, sin requirements/tests/licencia/semilla) | **Corregido y verificado** | `requirements.txt`, `LICENSE`, `codigo/app.py` con `debug=False` por defecto, semilla fija en todos los scripts de `codigo/experimentos/` |
| F19 | Sin procesamiento por lotes multi-SKU | **Corregido y verificado** | `codigo/forecasting_core/batch.py`, `batch_cli.py`; `codigo/tests/test_batch_memory.py` (memoria acotada con `tracemalloc`) |
| F20 | Sin intervalos de predicción ni derivación de inventario | **Corregido y verificado** | `codigo/forecasting_core/intervals.py`, `codigo/forecasting_core/inventory.py`; `codigo/tests/test_batch_memory.py` (pruebas de intervalos y política) |
| F21 | Horizonte sin restricciones, sin piso en cero, sin intervalo | **Corregido y verificado** | `ModelSpec.forecast(..., clip_non_negative=True)` aplicado idénticamente en backtest y pronóstico publicado |
| F22 | Filtrado estructural afirmado en el paper, ausente en el código | **Decisión de diseño aplicada: (a) implementado** | `codigo/forecasting_core/models.py::eligible_specs`; manuscrito Sección 2.3 documenta el mecanismo real, con cada exclusión y su motivo |
| F23 | 5 de 11 fórmulas mal tipografiadas en la tesis | **Corregido y verificado** | Manuscrito Sección 2.5 usa notación LaTeX estándar verificada contra `codigo/forecasting_core/metrics.py`; las fórmulas de la tesis (documento separado) no se corrigieron porque no forman parte del alcance de este repositorio de código |
| F24 | Inconsistencias bibliográficas y atribuciones incorrectas | **Corregido y verificado** | 33 entradas reales en `manuscritos/articulo_mdpi/template.tex`, formato numérico ACS; `check_cites.py`: 0 citas rotas, 0 bibitems huérfanos |
| F25 | Tabla de tiempos no mide el módulo dominante; medición manual, n=1 | **Corregido y verificado** | `codigo/experimentos/benchmark_tiempos.py` (`time.perf_counter()`, 5 repeticiones, incluye tuning); manuscrito Tabla 4 |

\* F07: la reconstrucción del `.tex` fue verificada estructuralmente (llaves
balanceadas, entornos `\begin`/`\end` correctamente anidados, columnas de
tablas consistentes con su especificación, todas las figuras referenciadas
existen en disco, 0 citas rotas). **No se pudo compilar a PDF real**: este
entorno de ejecución no tiene ninguna distribución LaTeX instalada
(`pdflatex`/`latexmk` no disponibles) y no se instaló una para evitar una
descarga/instalación de sistema no autorizada explícitamente. Los autores
deben compilar localmente (`pdflatex template.tex` dos veces + `bibtex` si
aplica, dentro de `manuscritos/articulo_mdpi/`) antes de enviar.

## Resultados cuantitativos finales

Todos reproducibles con un solo comando (ver `README.md`).

| Script | Resultado clave |
|---|---|
| `montecarlo_clasificacion.py` (1000 réplicas) | Falsos positivos tendencia: 74.4%→8.9% (media), 15.8% (máx). Estacionalidad: 50.2%→1.4% (media), 3.5% (máx). Potencia estacionalidad: 100%. |
| `caso_ilustrativo.py` | Serie de 36 meses con tendencia+estacionalidad: SARIMA gana con MASE=0.486 (46% mejor que seasonal naive, 48% mejor que naive); todos los métodos evaluados sobre n=8 orígenes idénticos. |
| `vs_incumbente.py` (40 series sintéticas, protocolo corregido con `honest_outer_estimate`) | Mejora mediana de MASE vs. incumbente: +19.9%. Supera al incumbente en 80% de series, al naive en 60%. MASE mediano: incumbente=0.922, naive=0.949, herramienta=0.807. Ver sección "Corrección post-hoc" abajo para el porqué del cambio frente a la primera corrida (que daba +0.9%/53%/50% por sesgo circular). |
| `panel_publico.py` (150 series M3-Monthly truncadas, protocolo corregido) | 150/150 series con ranking válido, sin exclusiones. La herramienta supera al naive en 63% de las series (vs. el 100% trivial del bug de circularidad). Wilcoxon signed-rank: W=2127.0, p<0.001 (significativo). Métodos clásicos ganan más a menudo en general (49%) que los automatizados ARIMA/SARIMA/ETS/Theta (34%), pero la automatización domina específicamente en los regímenes estacionales (43-64% de victorias vs. 21-23% en no estacionales) — hallazgo coherente con la competitividad de métodos simples documentada en la literatura M-competitions, y más interesante que la afirmación genérica que había generado automáticamente en el primer borrador ("la automatización domina"), que de hecho contradecía sus propios datos y fue corregida antes de publicar. |
| `benchmark_tiempos.py` (5 repeticiones) | n=24: 3.9s. n=48: 11.8s. n=72: 15.2s. n=96: 29.9s. n=120: 33.3s. Exponente empírico: 1.34 (vs. ~2.0 del original). Pipeline original completo (con tuning): 183.05s en n=120. |

## Corrección post-hoc durante la verificación

Al revisar los primeros resultados de `panel_publico.py` (150 series) antes
de publicarlos en el manuscrito, se encontró que **la herramienta "superaba
al naive" en el 100% de las series** — una cifra sospechosamente perfecta.
El diagnóstico: comparar el ganador de `run_pipeline` contra `naive` sobre el
mismo bloque de evaluación que decidió al ganador es circular, porque `naive`
es uno de los candidatos sobre los que se toma el argmin de MASE; el ganador
nunca puede tener peor MASE que `naive` en ese bloque por construcción
matemática, no porque pronostique mejor. Es la misma clase de sesgo de
selección que este refactor corrige para hiperparámetros (F05), aplicado esta
vez a la elección del *método*.

Se detectó el mismo problema, más un error de escala (el MASE del ganador se
escala internamente por `m_eff`, mientras la comparación original usaba
`m=12` fijo para incumbente/naive — números en escalas distintas para series
no estacionales), en `vs_incumbente.py`.

Ambos se corrigieron usando `honest_outer_estimate()` (`codigo/forecasting_core/optimize.py`),
que reserva un bloque **externo** que ni la selección del método ni el ajuste
de hiperparámetros vieron. Se encontró y corrigió además un `ValueError` de
pandas (índice duplicado cuando el ganador interno es exactamente `naive` o
`seasonal_naive`), con su propia prueba de regresión en
`codigo/tests/test_hyperopt_no_bias.py::test_estimacion_externa_no_duplica_el_modelo_si_el_ganador_es_naive`.

Este hallazgo — encontrado auditando mi propio trabajo, no el original — está
documentado aquí en vez de silenciado porque es exactamente el estándar de
transparencia que el resto de este refactor exige.

## Qué NO se hizo (alcance explícitamente fuera de este refactor)

- No se agregó machine learning ni deep learning (instrucción explícita del
  prompt maestro: el régimen de 24-48 observaciones no lo sostiene).
- No se paró el barrido de hiperparámetros en un tope arbitrario sin medir
  primero: se midió, y `statsforecast` resolvió el problema para ARIMA/SARIMA
  sin necesidad de barrido manual (F13).
- No se optimizó código fuera de lo que la Fase 7 identificó como cuello de
  botella real.
- No se corrigieron las fórmulas matemáticas del documento de tesis
  (`TESIS!.docx`) ni del PDF de la plantilla original — ambos son documentos
  separados del código y del manuscrito `.tex`; solo el manuscrito publicado
  (`manuscritos/articulo_mdpi/template.tex`) está dentro del alcance de este repositorio.
- No se validó contra los datos reales de la empresa de referencia (no disponibles en este
  entorno). `codigo/experimentos/vs_incumbente.py --input <archivo>` es la ruta
  directa para hacerlo cuando el archivo esté disponible.
- No se compiló el manuscrito a PDF (sin distribución LaTeX en este entorno;
  ver nota en F07 arriba).
- No se hizo `git commit` ni `git push`: todos los cambios quedan en el árbol
  de trabajo para que el usuario revise el diff antes de confirmar.
