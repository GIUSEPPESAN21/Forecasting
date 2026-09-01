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
  aceptar el Excel real de la empresa de referencia (`--input`); corre sobre un dataset
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

## Fase 11 — Comparación externa (Prophet / LightGBM), gráficos, manual e interactividad

Fase aditiva sobre el refactor 2.0.0: instrucción directa de los tutores del
proyecto (25/08/2026), documentada en `docs/prompt_maestro.md` de esta fase.
No reabre ni edita F01-F25; la única excepción declarada de antemano
(hook en `optimize.py`) resultó innecesaria y se documenta por qué en
`codigo/external_baselines/adapters.py`.

- **F26** — `comparacion_herramientas.pdf` (adjunto por los tutores) mezclaba
  el MAPE walk-forward interno de la Herramienta con métricas de Prophet
  sobre un holdout aparte, con `yearly_seasonality` de Prophet activada de
  forma incondicional incluso en n=24 (MAPE=353.98%, pronósticos negativos).
  Documentado en `codigo/experimentos/comparativa_externa.py` (docstring) y
  `codigo/experimentos/decision_prophet.md`. Verificado por
  `codigo/tests/test_external_baselines.py::TestProphetAdapter::test_yearly_seasonality_threshold_documented`.
- **F27** — Nuevo paquete `codigo/external_baselines/` (Prophet, LightGBM vía
  `mlforecast`), HERMANO de `forecasting_core/`, con imports perezosos y
  dependencias propias en `requirements-external.txt`. `PROPHET_SPEC`/
  `LIGHTGBM_SPEC` (`external_baselines/adapters.py`) son instancias REALES de
  `forecasting_core.models.ModelSpec`, no un duck-type, y se pasan sin
  ninguna modificación a `forecasting_core.validation.walk_forward`/
  `backtest_one_step` — el "hook aditivo" que el prompt de la Fase 11
  anticipaba para `optimize.py` resultó innecesario (razón documentada en el
  docstring de `adapters.py`) porque `comparativa_externa.py` evalúa Prophet/
  LightGBM directamente sobre el bloque EXTERNO que `honest_outer_estimate`
  ya reservó para la Herramienta, el mismo patrón que
  `experimentos/vs_incumbente.py` ya usa para el incumbente.
  `forecasting_core/optimize.py` **no se modificó**. Nuevo experimento
  `codigo/experimentos/comparativa_externa.py`: 10 longitudes (24-180) x 5
  regímenes estructurales, protocolo único de tres bloques para los tres
  métodos, salida en `resultados/comparativa_externa.csv` +
  `resultados/logs/comparativa_externa.log`. Verificado por
  `codigo/tests/test_external_baselines.py` (12 pruebas: forma/finitud del
  pronóstico, piso de no-negatividad F21 duplicado correctamente, historia
  insuficiente, no-fuga temporal con perturbación programática — mismo
  criterio que `test_no_leakage.py` — y compatibilidad `ModelSpec` real con
  `walk_forward`); se saltan automáticamente si `prophet`/`mlforecast`/
  `lightgbm` no están instalados (`pytest.importorskip` por clase, vía
  fixture `autouse`, no en el cuerpo de la clase, para que la ausencia de un
  solo paquete no aborte la colección de las demás clases del archivo).
- **F28** — Nuevo `codigo/experimentos/make_figures_comparativa.py` (archivo
  separado de `make_figures.py`, que no se toca): `fig_c1_boxplot_mase.png`
  (distribución de MASE por método), `fig_c2_mase_vs_longitud.png`
  (precisión vs. longitud de serie, un color por método),
  `fig_c3_panel_regimenes.png` (pequeños múltiplos: histórico + pronóstico
  de los tres métodos a la vez, 4 regímenes representativos — formato dual
  inspirado en el PDF de los tutores, pero con los tres métodos juntos y
  protocolo correcto). Tabla de reproducibilidad actualizada en `README.md`.
- **F29** — `docs/MANUAL_USUARIO.md`: manual de uso completo (instalación,
  los cinco módulos, exportación, `batch_cli.py`, preguntas frecuentes), sin
  requerir lectura de código. Modo demo en `codigo/app.py` Módulo 1: botón
  "Cargar datos de ejemplo" (`cargar_datos_demo`, un callback adicional con
  `allow_duplicate=True` sobre los mismos cuatro `Output` que
  `validar_y_mostrar`; ambos comparten el render vía `_procesar_carga()`,
  extraído sin alterar la lógica de `load_series`). Módulo 5 nuevo en
  `codigo/app.py`, "Comparación externa (Prophet / LightGBM)": import guard
  con `external_baselines.PROPHET_AVAILABLE`/`LIGHTGBM_AVAILABLE` (probados
  con `importlib.util.find_spec`, sin importar los paquetes pesados de
  verdad, para no gastar el presupuesto de RAM de la sesión interactiva si
  el módulo está deshabilitado); si ninguno está instalado, la sección
  muestra un aviso y el callback ni siquiera se registra. Los Módulos 1-4
  existentes no cambiaron de comportamiento, orden ni callbacks — verificado
  importando `app.py` con y sin los paquetes opcionales simulados presentes/
  ausentes (`importlib.util.find_spec` monkeypatcheado) antes de esta
  entrada del changelog.
- `requirements-external.txt` (nuevo, versiones fijadas: `prophet==1.4.0`,
  `cmdstanpy==1.3.0`, `mlforecast==1.1.0`, `lightgbm==4.7.0`) — NO se agregó
  nada a `requirements.txt` ni a `MODEL_REGISTRY` (restricción explícita del
  prompt de la Fase 11, §6).
- No regresión: la suite `pytest` original (ver conteo en
  `RESUMEN_EJECUCION.md`) sigue en verde sin cambios; las 12 pruebas nuevas
  de `test_external_baselines.py` son la única adición a `codigo/tests/`.

### Notas de reconciliación
- Ambas auditorías independientes, ejecutando el código original con
  versiones de librería ligeramente distintas, obtuvieron tamaños de
  parrilla idénticos (SES=19, Holt=361, HW=125+125, ARIMA=18, SARIMA=144) y
  el mismo resultado cualitativo en F01 (0% de casos satisfacen
  `es_muy_lineal`). `requirements.txt` fija las versiones usadas en este
  refactor para que el próximo auditor no tenga que reconciliar nada.

## Fase 12 — Correcciones de la revisión externa de tutores (2026-08-31)

Revisión externa de los tutores del proyecto: 2 bugs de código que invalidaban
cifras ya impresas en `manuscritos/articulo_mdpi/template.tex`, 3 piezas de
evidencia empírica faltantes, y un listado de problemas de redacción,
encuadre narrativo, formato MDPI y bibliografía. Ver
`RESUMEN_EJECUCION_FASE12.md` para el estado final detallado de cada
hallazgo con su cifra antes/después.

### Parte A — Código
- **F31** — `codigo/forecasting_core/intervals.py`: sigma y el ancho de banda
  de predicción podían angostarse con el horizonte por ruido de muestreo en
  horizontes con pocos orígenes (evidencia: sigma 360,347,192,204,99,31,57,
  65,24,1140,1195,1248 en `caso_ilustrativo_pronostico.csv`). Se fuerza
  `sigma_h = max(sigma_empirico_h, sigma_1*sqrt(h), sigma_{h-1})` y la misma
  regla sobre el ancho total de banda. Prueba: `codigo/tests/test_intervals.py`.
  Regenerado: sigma ahora 360→1248 monótono.
- **F32** — `codigo/experimentos/panel_publico.py::evaluate_one`: cuando el
  ganador era `naive` (o `seasonal_naive`), `mase_naive` (o
  `mase_seasonal_naive`) nunca se escribía — quedaba NaN, descartando 17 de
  150 series de los agregados por régimen (el Wilcoxon en sí, que ya excluía
  NaN por `dropna`, no estaba sesgado; los agregados por régimen y el
  desglose W/T/L sí). Corregido: columnas independientes. Prueba:
  `codigo/tests/test_panel_publico_mase_naive.py`. Rerun verificado:
  95 victorias/17 empates/38 derrotas (63%/11%/25%), Wilcoxon W=2127.0
  p<0.001 n=133 — confirma la cifra ya citada, ahora con el desglose
  completo y los medianas de régimen correctas (antes con hasta 9 filas
  faltantes por régimen).
- **F33** — `codigo/experimentos/ablacion_filtro_estructural.py` (nuevo):
  panel de 150 series con `structural_filter=True/False`. Prueba:
  `codigo/tests/test_ablacion_filtro_estructural.py`. Resultado: ver
  `RESUMEN_EJECUCION_FASE12.md`.
- **F34** — `codigo/experimentos/vs_incumbente.py`: Wilcoxon
  herramienta-vs-incumbente (`wtl_breakdown()`, extraída como función pura)
  y desglose W/T/L vs. naive. Prueba: `codigo/tests/test_vs_incumbente_wtl.py`.
  Rerun verificado (`--synthetic --n-series 40 --seed 20260824`): W=94.0
  p=4.93e-6 n=40; WTL vs. naive 24/10/6.
- **F35** — `codigo/forecasting_core/metrics.py::mase`: docstring precisa que
  el denominador escala sobre `scale_train` (bloque de entrenamiento,
  `y[:origins[0]]`), m=12 solo si se confirmó estacionalidad. Sin cambio de
  comportamiento.
- **F36** — `panel_publico.py` escribe `panel_publico_len{max_len}.csv` por
  corrida (alias `panel_publico.csv` para el caso base); nuevo
  `comparar_longitudes_panel.py`. Corridas a 24/36/48 (misma muestra,
  semilla 20260824): n=24 → 0/150 series (protocolo de tres bloques
  insatisfacible); n=36 → 150/150, MASE mediano 0.820 vs. naive 0.851, 50%;
  n=48 → 150/150, 0.701 vs. 0.829, 63%.
- **F37** — `montecarlo_clasificacion.py`: columnas `tipo_tendencia`/
  `tipo_estacionalidad` (tamaño/potencia) y tabla de potencia en `report()`.
  Los datos de potencia ya existían en `resultados/montecarlo_clasificacion.csv`
  (generadores con tendencia verdadera a n=24/36/48/120, sin cambio de
  lógica): potencia de tendencia en n=24 = 42.7% (lineal), 44.2% (deriva),
  0% (tendencia+estacional); n=36: 91.4%/55.3%/80.8%.
- **F38** — `codigo/forecasting_core/inventory.py`: docstrings de
  `safety_stock`/`compute_policy` precisan que `z` es un cuantil normal
  estándar y solo `sigma_L` es empírico. Sin cambio de comportamiento.
- **F39** — `codigo/experimentos/sensibilidad_outer_block.py` (nuevo):
  `outer_block` en {6,9,12}. Resultado: ver `RESUMEN_EJECUCION_FASE12.md`.
- Fix adicional (no ligado a un F3X): `panel_publico.py` y
  `comparar_longitudes_panel.py` ya no lanzan `KeyError`/`Traceback` cuando
  un `--max-len` deja 0 series con ganador (caso real: n=24) — reportan el
  motivo y salen limpio.
- **Verificación de la Parte A**: `pytest codigo/tests` — 266 pruebas antes
  (265 passed, 1 skipped) → 276 después (275 passed, 1 skipped, exit 0),
  incluyendo las 10 pruebas nuevas de F31-F34.

### Parte B — Reencuadre y anonimización del manuscrito
- **F40** — Introducción reestructurada en 3 contribuciones explícitas.
  Nueva Sección 3.7 "Measured Effect of Common Validation Pitfalls"
  (`sec:pitfalls`) consolida las comparaciones antes/después dispersas.
  Frases de encuadre de auditoría eliminadas o reescritas en Métodos/
  Resultados/Discusión/Conclusiones/GenAI/disponibilidad de datos/nota de
  bibliografía.
- **F41** — Título reutilizado de la decisión previa del usuario. 0
  ocurrencias del nombre real de la empresa en el `.tex` (`grep -ic` verificado).
  Introducida una sola vez como "a plastics manufacturing company
  (hereafter, the reference company)". Abstract reescrito, 200 palabras
  exactas (`wc -w` verificado sobre el bloque `\abstract{}`), sin prometer
  "evidence from an industrial case study" sin sustento cuantitativo.
  Declaración de revisión institucional alineada con la realidad (panel
  sintético, no registros reales agregados). Nota interna sin terminar en
  la Sección 3.4 eliminada. No se encontró ningún archivo real de datos de
  la empresa en el repositorio ni en disco (búsqueda verificada) — se aplicó
  la opción por defecto del prompt en su totalidad.

### Parte C — Contenido técnico con cifras verificadas
Cada cifra se verificó contra el CSV/log real antes de escribirse; ninguna
se inventó. Ver `RESUMEN_EJECUCION_FASE12.md` para la tabla completa de
verificación.
- **F42** — Sección 3.6 reescrita como "Comparison Against External
  Baselines (Prophet, LightGBM)" con `resultados/comparativa_externa.csv`
  (ya existente, no afectado por la Parte A): herramienta 0.747, Prophet
  0.785, LightGBM 0.893, naive 1.070 mediano global; 64%/72% victoria;
  LightGBM gana en n=24 (0.798 vs. 0.909) y n=120 (0.388). Prophet 1.4.0,
  LightGBM 4.7.0 vía mlforecast 1.1.0 (versiones verificadas con
  `python -c "import prophet, lightgbm, mlforecast"`).
- **F43** — Párrafo de la Figura 2 reescrito tras F31 (rerun de
  `caso_ilustrativo.py`): sigma 360→1248 monótono, ya no "widens sharply
  after month 9". Figura regenerada.
- **F44** — Wilcoxon panel público con desglose completo (ver F32). Tabla 4
  corregida: mediana de MASE naive en "seasonal, no trend" (0.962→0.940) y
  "flat" (0.722→0.783, ahora marginalmente MEJOR que naive en mediana pero
  con tasa de victoria de solo 45%, 25/56) tras poblar `mase_naive`/
  `mase_seasonal_naive` para las filas donde ganó el benchmark
  correspondiente.
- **F45** — Ablación del filtro estructural (ver F33): 150 series
  idénticas, `structural_filter=True/False`. MASE mediano 0.701 (on) vs.
  0.715 (off); victoria vs. naive 63.3% vs. 62.7%. Respuesta a la pregunta
  de investigación de la Introducción: el filtro aporta una mejora pequeña
  pero consistente (~2% relativo); la mayor parte de la ventaja de la
  herramienta sobre naive viene del resto del protocolo, no del filtro en
  sí.
- **F46** — "19.9%" aclarado como mediana de mejoras por serie; "12.5%"
  (mejora de medianas) como dato complementario explícito. Wilcoxon
  herramienta-vs-incumbente W=94.0 p=4.9e-6 agregado. "60%" vs. naive
  desglosado: 24 victorias/10 empates/6 derrotas.
- **F47** — Ecuación 1 (MASE) y texto circundante alineados con el
  docstring de F35 (denominador sobre `scale_train`, m=12/1 según
  estacionalidad confirmada); frase ambigua eliminada.
- **F48** — Sensibilidad de `outer_block` (ver F39): 150 series,
  `outer_block` en {6,9,12}. MASE mediano 0.701/0.707/0.739; victoria vs.
  naive 63.3%/71.3%/59.3% — patrón NO monótono, confirma la advertencia de
  varianza muestral de la Sección 2.5 sobre un MASE estimado con pocos
  orígenes. Seis orígenes se mantiene como valor por defecto en todo el
  paper por ser el menor común soportado por las tres longitudes del panel,
  no por producir el resultado más favorable (de los tres valores, ni el
  más alto ni el más bajo).
- **F49** — Panel M3 corrido a tres longitudes reales (ver F36); limitación
  de dominio explícita en Discusión (M3-Monthly no es demanda industrial;
  24-35 obs sin validar con series reales).
- **F50** — Potencia del test de tendencia (ver F37) agregada a
  Limitaciones. Abstract/Conclusiones: "single digits" → "8.9% mean (max
  15.8%)" con la cifra real de `montecarlo_clasificacion.csv`.
- **F51** — Stock de seguridad reescrito como SS=z·sigma_L (z cuantil
  normal, sigma_L empírico); aclarada la diferencia entre los 8 orígenes de
  la Tabla 2 y los 10 orígenes internos de `compute_policy`.
- **F52** — Terminología unificada a "Cochrane–Orcutt" (antes también
  "GLSAR(1)"). Hardware consolidado en una sola descripción (AMD Ryzen 5
  7000-series, 8 CPUs lógicos, 8GB RAM, Windows 11 build 26200 — verificado:
  el "Windows 10" previo era un artefacto conocido de `platform.platform()`
  en el mismo build 26200 que esta máquina). Objetivo de 25s aclarado como
  aplicable solo a n≤48. "183s" del pipeline original marcado como medición
  única, no comparable bajo el protocolo de 5 repeticiones de la Tabla 5.
  SD alta en n=96 documentada como límite de la medición (5 repeticiones,
  no reejecutado por presupuesto de tiempo).

### Parte D — Higiene mecánica y bibliografía
- **F53** — `\citeauthor{kerkkanen2009}` (incompatible con bibliografía
  numérica) → texto fijo "Kerkkänen et al. [10]". Referencias cruzadas
  Table~N/Figure~N escritas a mano → `\ref{}` (auditoría automática:
  0 refs sin `\label{}` correspondiente, 0 cites sin `\bibitem`
  correspondiente — script de verificación en el resumen de ejecución, sin
  `pdflatex` disponible en esta máquina para compilar). Autorreferencia de
  Sección 2.6 a sí misma corregida. Figura 1 regenerada: dos cajas "4."
  duplicadas → 9 etapas únicas, con el bloque externo del protocolo de tres
  bloques ahora explícito (antes ausente). Sección "Patents" (sin patentes)
  eliminada. Nota de ORCID recortada a una línea. Notas internas en español
  eliminadas. Declaración de uso de GenAI suavizada ("assisted in
  implementing" en vez de "executed the reproducibility scripts that
  produced the quantitative results"). Tabla de familia ganadora separa
  naive/seasonal naive en columnas propias (antes "Benchmark (naive)"
  mezclaba ambos).
- **F54** — Bibliografía verificada con WebSearch (35 entradas, comentario
  corregido de "33" a "35"): `talagala2021` → J. Forecast. 2023, 42,
  1476–1501 (antes working paper Monash 2021); `mentzer2001` → Moon,
  Mentzer & Smith, Int. J. Forecast. 2003, 19, 5–25 (antes J. Bus. Forecast.
  2001, 20, 5–11, journal/año/volumen/páginas incorrectos); `chopra2021` →
  7.ª ed. Pearson 2019 (antes 2021); `maack2024` → Vis. Comput. 2025, 41,
  1485–1498, publicado online 2024; `ollechwebel2020` → versión publicada
  J. Econom. Methods 2023, 12, 117–130 (no "Empirical Economics" como
  sugería el prompt original — verificado que el articulo se publicó en
  Journal of Econometric Methods, no en Empirical Economics; antes
  Deutsche Bundesbank Discussion Paper 2020). Agregadas 2 referencias
  nuevas para F42: `taylorletham2018` (Prophet, Am. Stat. 2018, 72, 37–45)
  y `garza2022mlforecast` (mlforecast, Nixtla).

## Fase 13 — Anonimización repo-wide (2026-09-01)

Una auditoría independiente posterior a la Fase 12 confirmó que
`manuscritos/articulo_mdpi/template.tex` ya estaba anonimizado, pero que el
nombre real de la empresa ("Tuboplex") seguía presente en 13 archivos del
repositorio público al que remite el Data Availability Statement del
manuscrito. Esta fase cierra esa fuga. Cambio puramente textual: ninguna
línea de lógica de negocio, algoritmos o datos numéricos fue modificada.

- **README.md** (tipo b + a): título anonimizado (título de documento,
  como `app.title`); referencias narrativas restantes reemplazadas por
  "la empresa de referencia"; ancla del enlace "Volver al inicio" ajustada
  al nuevo título.
- **codigo/app.py** (tipo b): comentario de cabecera y `app.title`
  ("Motor de Pronosticos - Tuboplex" → "Motor de Pronosticos"): referencia a
  la empresa eliminada, no reemplazada por una frase larga (título de
  pestaña del navegador).
- **docs/MANUAL_USUARIO.md** (tipo b): título del documento anonimizado.
- **codigo/experimentos/caso_ilustrativo.py** (tipo a): dos referencias en
  docstring reemplazadas.
- **codigo/experimentos/comparativa_externa.py** (tipo a): una referencia en
  comentario reemplazada.
- **codigo/experimentos/decision_prophet.md** (tipo a): una referencia
  narrativa reemplazada.
- **codigo/experimentos/vs_incumbente.py** (tipo a): cinco referencias
  (docstring, help de `--input`, mensajes impresos) reemplazadas.
- **resultados/logs/vs_incumbente.log** (tipo c): regenerado corriendo
  `vs_incumbente.py --synthetic --n-series 40 --seed 20260824` (mismos
  parámetros citados en la Tabla~\ref{tab:incumbente} del manuscrito) en vez
  de editado a mano. Todas las cifras citadas en el manuscrito se verificaron
  idénticas antes/después (MASE 0.922/0.949/0.807, MAPE 14.4/11.4/10.1,
  |ME| 217.1/55.4/111.9, 80%, 24/10/6, mejora +19.9%/+12.5%, Wilcoxon
  W=94.0 p=4.93e-06 n=40); `resultados/vs_incumbente.csv` y
  `vs_incumbente_resumen.csv` no cambiaron (sin diff tras la corrida).
- **docs/prompt_maestro.md** (tipo a): seis referencias (título del
  documento y cinco en prosa) reemplazadas.
- **docs/prompt_maestro_fase11.md** (tipo a): una referencia reemplazada.
- **RESUMEN_EJECUCION.md** (tipo a): dos referencias reemplazadas.
- **CHANGELOG.md** (tipo a): una referencia narrativa (entrada F09)
  reemplazada; una referencia dentro de un comando `grep -ic` citado en la
  entrada F41 reformulada sin nombrar la empresa, preservando el
  significado histórico del check (0 ocurrencias) sin reintroducir la fuga.
- **RESUMEN_EJECUCION_FASE12.md** (tipo a): dos referencias dentro de
  comandos `grep -ic` citados (entradas F40/F41) reformuladas de la misma
  manera que en CHANGELOG.md.

Verificación (ver `RESUMEN_EJECUCION_FASE13.md` para la salida completa de
cada comando): `grep -rli "tuboplex" .` → vacío; `grep -ic "tuboplex"
template.tex` → 0 (sin cambios); `pytest codigo/tests -q` → 276 tests, 0
fallos, 0 errores (el conteo de skipped difiere del de la auditoría por
disponibilidad de paquetes opcionales — prophet/mlforecast/lightgbm — en
este entorno, no por los cambios de esta fase).

git history nota: el nombre real de la empresa sigue presente en commits
anteriores a esta fase. Reescribir el historial de git está fuera de
alcance de esta fase (ver discusión en `RESUMEN_EJECUCION_FASE13.md`).
