# Changelog

Refactor completo del motor de pronósticos siguiendo el prompt maestro que
fusiona dos auditorías independientes (25 hallazgos, tabla de referencia
cruzada en el prompt original). Cada entrada cita el ID de hallazgo que
resuelve y el archivo donde vive la corrección.

## [2.0.0] — 2026-08-24

### Fase 0 — Arquitectura
- Se creó `codigo/forecasting_core/`, paquete puro sin dependencias de Dash/Plotly,
  testeable de forma independiente. `codigo/app.py` pasó a ser una capa delgada que
  solo construye el layout y conecta callbacks.
- `tesisVF.py` (eliminado del arbol de trabajo tras el refactor; recuperable via `git show 63bf54e:tesisVF.py`) (1301 líneas, monolito Dash+lógica) queda sustituido por
  `codigo/forecasting_core/` + `codigo/app.py`.

### Fase 1 — Núcleo estadístico
- **F01** — `es_muy_lineal()` (predicado insatisfacible: `R²≥0.90` y
  `|ACF₁₂|<0.10` simultáneos, 0% de casos satisfechos en 3000 series
  sintéticas) reemplazado por `allow_constant_level_methods()` en
  `codigo/forecasting_core/classification.py`: nivel constante se admite cuando la
  serie no tiene tendencia significativa **y** es estacionaria en nivel.
- **F04** — `_make_predictor()` (despacho por subcadena, fallback silencioso a
  `train[-1]` para nombres no reconocidos) reemplazado por
  `MODEL_REGISTRY` con despacho por clave exacta en
  `codigo/forecasting_core/models.py`; una clave desconocida levanta `KeyError`.
- **F10** — Detección de estacionalidad por `|ACF(12)|>0.30` (50.2% de falsos
  positivos sobre series con solo tendencia) reemplazada por
  `seasonality_test()`: fuerza estacional STL (Wang, Smith & Hyndman 2006)
  **y** significancia Kruskal-Wallis sobre residuos detrendizados, con mínimo
  de 3 ciclos completos.
- **F11** — Test de tendencia por p-value de OLS clásico (74.4% de falsos
  positivos sobre paseos aleatorios) reemplazado por el procedimiento
  secuencial de Dickey-Fuller/Pantula en `trend_test()`: ADF('ct') primero,
  luego GLSAR(1) sobre el nivel si es trend-stationary, o deriva de la
  primera diferencia con errores HAC si es I(1). ADF confirmado con KPSS.
- **F22 (decisión a)** — El filtrado por características estructurales que el
  manuscrito afirmaba y el código no implementaba ahora existe:
  `eligible_specs()` en `codigo/forecasting_core/models.py`, con cada exclusión
  registrada y su motivo.
- Orden corregido: estacionalidad se prueba primero, la serie se
  desestacionaliza (por medias estacionales sobre la serie detrendizada, no
  STL — ver docstring de `deseasonalize()`), y solo entonces se prueba
  tendencia. Probar tendencia sobre la serie cruda perdía la tendencia en el
  100% de los casos con estacionalidad fuerte y n=24.
- Validado con `codigo/experimentos/montecarlo_clasificacion.py` (1000 réplicas):
  falsos positivos de tendencia 74.4%→8.9% (media), de estacionalidad
  50.2%→1.4% (media), potencia de estacionalidad 100%.

### Fase 2 — Métricas
- **F06** — Mean Error (ME) implementado en `codigo/forecasting_core/metrics.py`
  (declarado en ambos manuscritos previos, nunca implementado).
- **F12** — `mape()` ahora excluye explícitamente periodos con demanda cero
  (antes: `max(|y|,1e-8)` producía valores como 12,500,000,003%). `mase()`
  añadido como métrica primaria de ranking (Hyndman & Koehler 2006).
- `tracking_signal()` añadido como diagnóstico de sesgo sistemático.

### Fase 3 — Validación honesta
- **F02/F14** — `walk_forward_errors`/`walk_forward_detail` (recorridos
  triplicados por sesión) fusionados en una sola función
  `walk_forward()` en `codigo/forecasting_core/validation.py` que devuelve agregado y
  detalle a la vez. Un método con cualquier origen fallido queda
  **excluido del ranking**, nunca promediado sobre menos puntos.
- **F02** — `min_train` ahora es `max(10, 2·m)` para todos los métodos
  estacionales, sin fallback silencioso de Holt-Winters→Holt ni de
  SARIMA→ARIMA(1,1,1) a mitad del walk-forward.
- **F03** — La fuga in-sample de `_fitted_series` (`rolling(k).mean()`
  incluía la observación que pronosticaba; MAPE mostrado 20.30% vs. honesto
  35.38%) eliminada por construcción: `codigo/forecasting_core/models.py` expone una
  única función `ModelSpec.forecast()` usada tanto por el backtest como por el
  pronóstico final, con `h=1` para el caso de un paso.
- **F05/F13** — `codigo/forecasting_core/optimize.py`: los orígenes del walk-forward
  se parten en bloque de *tuning* y bloque de *evaluación*, disjuntos y
  contiguos; ningún hiperparámetro ve el bloque de evaluación. Para
  ARIMA/SARIMA el barrido desaparece: se usa `AutoARIMA`/`AutoETS`/`AutoTheta`
  de `statsforecast` (selección por AICc). Parrillas de SES/Holt/Holt-Winters
  reducidas a coarse-to-fine (máx. 27 combinaciones vs. 361 originales).
  `honest_outer_estimate()` añadido para estimar desempeño sin sesgo también
  en la elección del *método* (no solo de sus hiperparámetros) — ver nota de
  corrección más abajo.

### Fase 5 — Intervalos, inventario, horizonte, lote
- **F20** — `codigo/forecasting_core/intervals.py`: intervalos de predicción por
  cuantiles empíricos del error de backtest por horizonte (con degradación
  documentada a aproximación normal si hay pocos orígenes).
- **F20/M-08** — `codigo/forecasting_core/inventory.py`: stock de seguridad y punto
  de reorden a partir de la desviación del error **acumulado** sobre el lead
  time (no `sigma_1·sqrt(L)`, que asume independencia entre periodos).
  Detecta y avisa cuando el sesgo acumulado domina sobre la variabilidad.
- **F21** — Piso de no-negatividad aplicado de forma idéntica en backtest y
  pronóstico publicado (`ModelSpec.forecast(..., clip_non_negative=True)`).
- **F19** — `codigo/forecasting_core/batch.py`: procesamiento multi-SKU con memoria
  acotada (volcado incremental a CSV, `n_jobs = min(4, cpu_count-2)`, nunca
  `-1`). Verificado en `codigo/tests/test_batch_memory.py` con `tracemalloc`.

### Fase 6 — Robustez de datos y manejo de errores
- **F15** — `codigo/forecasting_core/data.py`: acepta meses numéricos, en inglés y
  abreviados; detecta y declara la convención de separador decimal/miles;
  reporta duplicados (consolidando los idénticos, rechazando los
  contradictorios); nunca colapsa huecos temporales en silencio — exige una
  política explícita (`report`/`interpolate`/`zero`/`abort`).
- **F16** — Eliminado `warnings.filterwarnings("ignore")` global. Logging
  configurable vía `FORECASTING_LOG_LEVEL`. Excepciones específicas con
  contexto en cada módulo.

### Fase 7 — Medición de desempeño
- Reescrito con `time.perf_counter()`, 5 repeticiones, media±desviación,
  **incluyendo el módulo de optimización** (ausente en la Tabla 3 original).
  `codigo/experimentos/benchmark_tiempos.py`. Resultado: 33.3s en n=120 (vs. 183.05s
  medidos en el pipeline original completo, que el manuscrito reportaba en
  89.79s por omitir la optimización). Exponente empírico de complejidad
  1.34 (vs. ~2.0 del original).

### Fase 8 — Validación estadística ampliada
- **8.1** — `codigo/experimentos/montecarlo_clasificacion.py`: ver Fase 1.
- **8.2** — `codigo/experimentos/panel_publico.py`: validación sobre 150 series de
  M3-Monthly truncadas a ≤48 observaciones, con prueba de Wilcoxon
  signed-rank. **Corrección post-hoc**: la primera corrida comparaba el
  ganador de `run_pipeline` contra `naive` sobre el mismo bloque que decidió
  al ganador — circular, porque `naive` es uno de los candidatos sobre los
  que se toma el argmin de MASE, así que el ganador nunca puede perder ahí
  por construcción matemática (resultado: "100% de las series superan al
  naive" en las 150 series, detectado como sospechoso al revisar antes de
  publicar). Corregido usando `honest_outer_estimate()`: el ganador se evalúa
  sobre un bloque externo que ni la selección del método ni el ajuste de
  hiperparámetros vieron.
- **8.3** — Decisión documentada en `codigo/experimentos/decision_prophet.md`: la
  comparación con Prophet (sin protocolo, datos ni código en el manuscrito
  original) se retira y se reemplaza por la comparación contra
  `AutoARIMA`/`AutoETS`/`AutoTheta`, ya integrados sin costo de dependencia
  adicional.

### Fase 4 — Línea base
- **F09** — `codigo/experimentos/vs_incumbente.py`: comparación herramienta vs.
  método incumbente (promedio móvil k=3) vs. naive. Parametrizado para
  aceptar el Excel real de Tuboplex (`--input`); corre sobre un dataset
  sintético equivalente mientras esos datos no estén disponibles (ver
  limitación en el manuscrito, Sección 2.8). **Corrección post-hoc**: el
  mismo sesgo circular de 8.2, más un error de escala (el MASE del ganador se
  escalaba por `m_eff` interno mientras incumbente/naive se calculaban con
  `m=12` fijo, comparando números en escalas distintas para series no
  estacionales). Corregido con `honest_outer_estimate()` y escalamiento
  consistente extraído del mismo bloque externo.

### Fase 9 — Manuscrito
- **F07** — Eliminado el andamiaje de plantilla (`How to Use this Template`,
  bloques `ELIMINAR`, tabla/figura de ejemplo, apéndice de ejemplo).
  `\documentclass[journal,...]` corregido a `[forecasting,...]` (el nombre de
  journal nunca se había fijado). Autores, afiliación, correo y CRediT
  completados. Manuscrito movido a `manuscritos/articulo_mdpi/template.tex` con
  `manuscritos/articulo_mdpi/Definitions/` y figuras reales en `manuscritos/articulo_mdpi/figures/`.
- **F09 (figuras)** — Las 10 referencias a `flowchart_tool.png` (una imagen
  reutilizada 10 veces, con 10 `\label{fig:demanda}` idénticos) reemplazadas
  por 2 figuras reales generadas por `codigo/experimentos/make_figures.py`: un
  diagrama de flujo del pipeline corregido y el gráfico de pronóstico del
  caso ilustrativo, cada una con `\label` único.
- **F23** — Las 5 fórmulas mal tipografiadas de la tesis no se trasladaron al
  manuscrito: la Sección 2.5 (Error Metrics) del `.tex` usa notación LaTeX
  estándar, verificada contra `codigo/forecasting_core/metrics.py`.
- **F24** — Bibliografía reconstruida con 33 entradas reales, formato
  numérico ACS (`Forecasting` no usa cita autor-año), cada `\cite{}`
  verificado contra un `\bibitem` existente y viceversa
  (`check_cites.py`, 0 huérfanos). Atribuciones incorrectas del manuscrito
  original (Benidis et al. 2022 sobre grid search; Bergmeir et al. 2020 en
  vez de Bergmeir & Benítez 2012) no se repitieron.
- **F17** — Limitación de variables exógenas reformulada de forma específica
  (cartera adjudicada de proyectos de construcción) en vez de la mención
  genérica original.
- Declaraciones MDPI completadas con contenido real: Author Contributions
  (CRediT), Funding, Institutional Review ("Not applicable"), Informed
  Consent ("Not applicable"), Data Availability (repositorio con scripts
  reproducibles, datos sintéticos donde los reales son confidenciales),
  Acknowledgments, Conflicts of Interest, Abbreviations.
- Sección "Comparison with Prophet" retirada y reemplazada (ver Fase 8.3).
- Sección Conclusions escrita (antes: placeholder de la plantilla).

### Fase 10 — Higiene de repositorio
- **F18** — `requirements.txt` con versiones fijadas; `codigo/app.py` con
  `debug=False` por defecto (activable con `FORECASTING_DEBUG=true`);
  `_backtest_with_best` (código muerto, nunca invocado en el original) no
  se trasladó — su única función honesta quedó absorbida en
  `ModelSpec.forecast()`.
- `LICENSE` (MIT), `.gitignore`, `README.md` con instrucciones de ejecución
  de cada script de `codigo/experimentos/`.
- Limpieza de `__MACOSX/` y `__pycache__/`.
- Suite `pytest` (`codigo/tests/`): clasificación, métricas, ausencia de fuga
  temporal (verificada programáticamente), paridad del walk-forward,
  registro de modelos, ausencia de sesgo de hiperparámetros, carga de datos,
  memoria del lote e intervalos/inventario. Ver `RESUMEN_EJECUCION.md` para
  el conteo final.

### Notas de reconciliación
- Ambas auditorías independientes, ejecutando el código original con
  versiones de librería ligeramente distintas, obtuvieron tamaños de
  parrilla idénticos (SES=19, Holt=361, HW=125+125, ARIMA=18, SARIMA=144) y
  el mismo resultado cualitativo en F01 (0% de casos satisfacen
  `es_muy_lineal`). `requirements.txt` fija las versiones usadas en este
  refactor para que el próximo auditor no tenga que reconciliar nada.
