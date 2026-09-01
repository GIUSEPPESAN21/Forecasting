# PROMPT MAESTRO — Refactor y validación integral del motor de pronósticos de la empresa de referencia

> **Cómo usar este documento:** pégalo completo como instrucción inicial a un agente de codificación (Claude Code u otro) con acceso de escritura al repositorio `GIUSEPPESAN21/Forecasting` (o a una copia local en el PC de destino). El agente debe seguir las fases en orden — cada una es prerrequisito de la siguiente — y no debe declarar una fase terminada sin que sus pruebas de aceptación pasen. Este documento consolida **dos auditorías independientes** del mismo código (`tesisVF.py`, `TESIS!.docx`, `tesis pronosticos/template.tex`), 25 hallazgos verificados por ejecución real, sin duplicados, con ID único de referencia cruzada.

---

## 0. Rol y misión

Eres un ingeniero senior de *forecasting* con dominio simultáneo de estadística de series de tiempo, arquitectura de software en Python y planeación de inventarios. Tu misión no es "arreglar bugs sueltos": es **llevar `tesisVF.py` de un prototipo académico con seis defectos que invalidan sus propios resultados publicados, a un sistema de pronóstico correcto, honesto estadísticamente, ligero en cómputo y con evidencia empírica reproducible** — al nivel de rigor que exige un *Feature Paper* de la revista *Forecasting* (MDPI).

Trabajas contra el repositorio real. Cada corrección debe:
1. Citar el archivo y las líneas que modifica.
2. Incluir una prueba automatizada que falle con el código viejo y pase con el nuevo (regresión).
3. Quedar documentada en un `CHANGELOG.md` con el ID de hallazgo que resuelve (tabla en la §2).

## 1. Restricción de hardware — no negociable

El sistema final debe ejecutarse fluidamente en un **PC con AMD Ryzen 5 serie 7000 (6 núcleos / 12 hilos, boost ~5.1–5.4 GHz) y 8 GB de RAM total** (es decir, ~5–6 GB disponibles tras el sistema operativo y el navegador). Esto es un objetivo de ingeniería, no una sugerencia: cada decisión de arquitectura de las fases siguientes está calibrada para respetarlo. Presupuestos concretos:

| Escenario | RAM máx. residente | Tiempo objetivo |
|---|---|---|
| Sesión interactiva, 1 serie, n≤48 | ≤ 1.5 GB | Clasificación+ranking ≤ 5 s · optimización del ganador ≤ 15 s · forecast final ≤ 2 s (pipeline completo ≤ 25 s) |
| Lote multi-SKU, cualquier tamaño de portafolio | ≤ 3 GB (constante, sin crecer con el número de SKU — procesar y volcar a disco, nunca acumular en memoria) | ≤ 5 s promedio por SKU |
| Barrido de hiperparámetros de un método | Un solo proceso worker ≤ 400 MB | Acotado por número de *folds*, no por tamaño de la serie (ver Fase 3) |

Reglas de diseño que se derivan de esto:
- `n_jobs = min(4, os.cpu_count() - 2)` en cualquier paralelismo con `joblib` — nunca `-1`. Cada proceso worker que importe `statsmodels` cuesta ~150–300 MB solo en el import; no lances más de los que la RAM tolera.
- En procesamiento por lotes: **procesar serie por serie y volcar el resultado incrementalmente** (`parquet`/`csv` en modo *append*), nunca acumular un DataFrame con todo el portafolio en memoria.
- Preferir **`statsforecast` (Nixtla)** sobre `statsmodels` para la selección automática de ARIMA/ETS: es Numba-JIT, ~10–50× más rápido, con memoria mucho más acotada, y con `AutoARIMA`/`AutoETS` resuelve además el sesgo de selección de la Fase 3 (selección por verosimilitud/AICc, no por barrido exhaustivo de MAPE). Es la palanca única que más ayuda a cumplir "ligero pero potente".
- El barrido manual de hiperparámetros (`_grid`, `_tunar_mejor`) se conserva solo como *fallback* documentado para SES/Holt (baratos) y se **retira por completo para ARIMA/SARIMA**, sustituido por `AutoARIMA` de `statsforecast`.
- Nunca `warnings.filterwarnings("ignore")` global. Usar `logging` con nivel configurable; los `ConvergenceWarning` deben quedar en el log, no ocultos.

---

## 2. Registro maestro de hallazgos (fusión de ambas auditorías, sin duplicados)

`ID` es la referencia canónica que debes usar en commits, tests y `CHANGELOG.md`. `Auditoría A` = referato de este chat; `Auditoría B` = artefacto "Auditoría Motor de Pronósticos de la empresa de referencia" adjuntado por el usuario. `Fase` indica en qué fase de la §3 se resuelve.

| ID | Título | Auditoría A | Auditoría B | Fase |
|---|---|---|---|---|
| F01 | `es_muy_lineal()` insatisfacible (R²≥0.90 **y** \|ACF₁₂\|<0.10 son mutuamente excluyentes) — 0/3000 en simulación | C-01 | C-01 | 1 |
| F02 | Comparación no homogénea: n_preds distinto por método; "Holt-Winters"/"SARIMA" son en realidad Holt/ARIMA(1,1,1) durante 8–14 de los pasos por *fallback* silencioso de `min_train` insuficiente | C-02 | M-01 | 3 |
| F03 | Fuga in-sample en el "MAPE histórico ajustado" mostrado en el gráfico principal (`_fitted_series`, rolling incluye y[t]); la función honesta `_backtest_with_best` existe y nunca se llama | C-03 | C-04 | 3 |
| F04 | `_make_predictor` despacha por subcadena y cae al *fallback* `train[-1]` (paseo aleatorio) para Promedio Simple/Móvil/Ponderado — el "MAPE optimizado" del Módulo 3 es el de un modelo distinto al reportado | C-04 | — (único de A) | 1 |
| F05 | Sesgo de selección: hiperparámetros elegidos sobre el mismo walk-forward cuyo MAPE se reporta como desempeño; además SES/Holt/HW compiten en el ranking inicial con α=β=γ=0.1 fijos frente a métodos sin parámetros | C-05 | C-05 | 3 |
| F06 | Error Medio (ME) declarado en ambos manuscritos como métrica calculada — no existe en el código | C-06 | M-05 | 2 |
| F07 | `template.tex` no compila (figura inexistente ×10), conserva texto de plantilla, `\Title{Title}`, autores/afiliación placeholder, **Conclusiones vacías**, declaraciones MDPI obligatorias sin completar, bibliografía de ejemplo | C-07 | C-09 | 9 |
| F08 | Sección "Comparison with Prophet": resultados cuantitativos narrados sin figura, sin tabla, sin protocolo, sin una línea de Prophet en el código | C-08 | Sección "Comparison with Prophet" | 9 |
| F09 | Afirmación central ("mejora significativamente la precisión") sin línea base: el error del método incumbente de la empresa de referencia nunca se calculó; sin benchmark ingenuo no hay forma de interpretar un MAPE aislado | C-09 | C-10 | 4 |
| F10 | Detector de estacionalidad \|ACF₁₂\|>0.30 sobre serie sin desestacionalizar: 100% falsos positivos sobre tendencia pura en simulación | M-01 | C-07 (parcial, con validación Monte Carlo: 50.2% falsos positivos) | 1 |
| F11 | Prueba de tendencia (p-value OLS no robusto a autocorrelación) y ADF con `regression="c"` mal especificado frente a series con tendencia; baja potencia con n=24 | M-02 | C-07 (Monte Carlo: 74.4% falsos positivos de tendencia sobre random walk; ADF con 18.3% falsos positivos, 51.7% potencia real) | 1 |
| F12 | MAPE exploza con demanda cero (eps=1e-8 no protege); sin benchmark naive/seasonal naive; sin MASE — en una serie estacional la herramienta pierde 54% contra seasonal naive | M-03 | C-06, M-04 | 2 |
| F13 | Límites de cómputo del grid search inconsistentes con lo declarado: HW acotado a 300 combos pero solo alcanza 250; Holt (361 combos) y ARIMA/SARIMA sin tope real | M-04 | Evidencia cuantitativa en C-05 (20× más combos que observaciones en Holt) | 3/5 |
| F14 | Cómputo redundante: walk-forward ejecutado 3–4 veces por sesión (Módulos 2 y 3); `cargar_serie()` reparseada 5 veces | M-05 | M-02 (medido: ~66% del tiempo de respuesta es cómputo duplicado) | 3 |
| F15 | Carga de datos frágil: falla con meses numéricos, meses en inglés, separador de miles, filas duplicadas; huecos temporales se colapsan en silencio y desalinean el índice estacional | M-06 | M-06 | 6 |
| F16 | 16 bloques `except Exception` sin registro; `warnings.filterwarnings("ignore")` global oculta fallos de convergencia de máxima verosimilitud que igual entran al ranking | M-07 | m-01 (último punto) | 6/10 |
| F17 | Limitación no declarada: demanda dirigida por proyectos de obra (variable exógena predictiva conocida y disponible) — más crítica que la limitación univariante ya declarada | M-08 | — (único de A) | 9 |
| F18 | Higiene de repositorio: `debug=True`, código muerto, sin `requirements.txt`/tests/licencia/semilla, constantes mágicas repetidas, todo el cómputo dentro de callbacks sin caché | m-09 | m-01 | 10 |
| F19 | Sin procesamiento por lotes multi-SKU (`multiple=False`); operación real de 200 referencias no soportada | M-14 | M-09 | 5 |
| F20 | Sin intervalos de predicción (contradice el marco teórico de *uncertainty-aware visual analytics* citado); no se deriva σ del error, stock de seguridad ni punto de reorden — la herramienta no cierra el ciclo hacia la decisión de inventario | M-15 | C-08, M-08 | 5 |
| F21 | Horizonte de pronóstico sin restricciones: proyecta 24 meses sobre 24 meses de historia con una recta de pendiente negativa, sin piso en cero y sin intervalo (caída del 64% presentada como insumo de decisión) | — (mencionado en general) | M-07 (único de B, con cifras) | 5 |
| F22 | El paper describe un mecanismo de *filtrado estructural por características* (cita a Montero-Manso et al. 2020, Talagala et al. 2021) que **no existe en el código** — es la afirmación de novedad central del manuscrito | — (único de B) | C-03 | 1 (decisión) |
| F23 | 5 de 11 fórmulas matemáticas de la tesis mal tipografiadas: índice de SES desplazado, paréntesis faltantes en (1-α)/(1-β)/(1-γ) de Holt y Holt-Winters, paréntesis faltantes en ARIMA, ecuación de SARIMA ausente, MAD y MAPE sin barras de valor absoluto (el código sí las calcula — documento y código divergen) | — (único de B) | Sección 04, tabla de ecuaciones | 9 |
| F24 | Inconsistencias bibliográficas: años de cita distintos entre secciones (Hyndman & Athanasopoulos 2018/2021, Maack 2024/2025), Nahmias (2007) citado sin existir en la lista, 11 citas del `.tex` sin entrada en bibliografía, atribuciones incorrectas (Benidis et al. 2022 es sobre *deep learning*, no *grid search*; Bergmeir et al. 2020 debería ser Bergmeir & Benítez 2012) | — (único de B) | Sección 04 | 9 |
| F25 | Tabla de tiempos computacionales no mide el módulo dominante: el Módulo 3 (optimización) es ~60% del tiempo total y no aparece en la Tabla 3; medición con cronómetro manual, n=1, sin especificar hardware | M-11 (parcial) | M-03 (medido: 183 s reales vs. 89.8 s reportados, n=120) | 7 |

**Nota de reconciliación:** ambas auditorías, ejecutando el mismo código con versiones de librería ligeramente distintas (statsmodels 0.14.6 en ambos casos; pandas 2.2.2 vs 3.0.2), obtuvieron tamaños de parrilla idénticos (SES=19, Holt=361, HW=125+125, ARIMA=18, SARIMA=144) y el mismo resultado cualitativo en F01 (0% de casos satisfacen `es_muy_lineal`). Esa coincidencia entre entornos es la confirmación más fuerte posible de que estos no son artefactos de una sola ejecución. Fija versiones en `requirements.txt` para que el próximo auditor no tenga que reconciliar nada.

---

## 3. Fases de ejecución (orden de dependencia — no saltar pasos)

### Fase 0 — Arquitectura base
Reestructura el repositorio antes de tocar lógica:
```
forecasting_core/          # paquete puro, CERO imports de dash/plotly
  data.py                  # carga y validación (acepta DataFrame directo, testeable sin Excel)
  classification.py        # tendencia / estacionalidad / estacionariedad
  models.py                # registro {nombre: ModelSpec(fit, predict, grid, min_obs, es_estacional)}
  metrics.py                # mse, mad, mape (zero-safe), me, mase
  validation.py            # walk-forward único (agregado + detalle en una sola pasada), CV anidada
  optimize.py               # AutoARIMA/AutoETS vía statsforecast + grid manual acotado para SES/Holt
  intervals.py              # intervalos de predicción por método
  inventory.py               # sigma del error -> stock de seguridad -> punto de reorden
  batch.py                   # orquestación multi-SKU con presupuesto de memoria constante
app.py                       # SOLO Dash: callbacks delgados que llaman a forecasting_core y cachean en dcc.Store
tests/                       # pytest, ver Fase 10
experiments/                 # scripts de validación empírica, ver Fase 8
manuscript/                  # .tex corregido, ver Fase 9
requirements.txt · README.md · LICENSE · CHANGELOG.md
```
Regla de oro que gobierna todo lo que sigue: **ninguna métrica se reporta jamás calculada sobre los mismos datos usados para elegir el modelo o sus hiperparámetros.**

### Fase 1 — Núcleo estadístico correcto (bloquea todo lo demás)
- **F01**: separar el predicado en dos pruebas independientes. Los métodos de nivel constante (Promedio Simple/Móvil/Ponderado) se habilitan cuando la serie **no tiene tendencia significativa** (p≥0.05 en el test robusto de abajo), no cuando "es muy lineal". Escribe el test de regresión con las series sintéticas ya usadas en ambas auditorías (`tend n=24 σ=10`, `tend PERFECTA n=24`, etc.) y verifica que el predicado ahora se activa correctamente.
- **F04**: sustituye el despacho por subcadena de `_make_predictor` por un registro explícito `MODEL_REGISTRY: dict[str, ModelSpec]` con clave exacta (no `in`). Cualquier nombre sin coincidencia exacta debe **lanzar excepción**, nunca caer a un modelo distinto en silencio.
- **F10 + F11**: reemplaza la detección de estacionalidad por fuerza estacional vía descomposición STL: `F_S = max(0, 1 − Var(residuo)/Var(estacional+residuo))` (Wang, Smith & Hyndman, 2006), umbral 0.3–0.6, calculada sobre la serie **detrendizada**. Reemplaza el test de tendencia por OLS con errores HAC/Newey-West (`sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags":1})`). Ejecuta `adfuller` con `regression="ct"` cuando el test de tendencia sea positivo, y añade KPSS como par confirmatorio (hipótesis nula opuesta a ADF — coincidencia entre ambos da la clasificación final). Añade un aviso explícito de baja potencia cuando n<48.
- **F22 — decisión de diseño, no aplazar**: el manuscrito afirma un filtrado por características que hoy no existe. Elige y documenta una de las dos opciones (no dejar la afirmación como está):
  - (a) **Implementarlo de verdad**: usar la clasificación ya corregida (F10/F11) para excluir candidatos incompatibles con la estructura detectada (ej. no evaluar HW/SARIMA si no hay estacionalidad con confianza suficiente), citando explícitamente Montero-Manso et al. (2020) y Talagala et al. (2021) como el mecanismo que sí se implementó; o
  - (b) **Retirar la afirmación** del `.tex` y reformular la contribución como comparación exhaustiva sin preselección (más simple, pero hay que reescribir la Introducción y el Abstract).
  Recomendación: (a), porque además reduce el costo computacional de la Fase 5 al no evaluar modelos que la propia clasificación descarta.

Prueba de aceptación de la fase: con las series sintéticas de control (plana, tendencial, estacional, tendencia+estacionalidad), el ganador que produce el pipeline debe coincidir con el ganador estadísticamente esperado en ≥90% de 200 repeticiones por escenario, y el reporte de Monte Carlo de F10/F11 debe mostrar tasas de falso positivo cercanas al nivel nominal (~5%), no al 50–74% medido hoy.

### Fase 2 — Métricas y benchmarks
- **F06**: implementa `me(y_true, y_pred)`. Añádelo a la tabla de métricas del Módulo 2 junto a una **señal de rastreo** (Σerror/MAD) — es la métrica de sesgo, la que decide si un error sistemático genera sobrestock o faltante.
- **F12**: añade `naive` (último valor) y `seasonal_naive` (valor de hace 12 meses) a `MODEL_REGISTRY` como candidatos evaluados siempre. Implementa `mase(y_true, y_pred, y_train, m=12 si estacional else 1)` y conviértela en la **métrica primaria de ranking**; conserva MAPE como métrica secundaria de comunicación, excluyendo explícitamente del cálculo los periodos con demanda cero (con una nota visible en la interfaz, no un `eps` que dispara el valor a miles de millones de por ciento).

Prueba de aceptación: con `y_true=[100,0,120,110]`, el MAPE reportado ya no puede ser un número de más de 4 dígitos; debe excluir el cero y quedar documentado. En la serie estacional sintética, seasonal naive debe aparecer en la tabla comparativa junto al resto — si gana, el ranking debe decirlo (hoy pierde 54% contra un candidato que ni siquiera se evalúa).

### Fase 3 — Validación honesta (el corazón del refactor)
- **F02 + F13 + F14**: reescribe `validation.py` con **una sola función** que recorra el walk-forward una vez y devuelva tanto las métricas agregadas como el detalle por origen (elimina la duplicación entre `walk_forward_errors`/`walk_forward_detail`/la re-llamada del Módulo 3 — F14). Fija `min_train = max(10, 2·m)` para **todos** los métodos, no solo los estacionales; si un método no puede evaluarse en un origen determinado, márcalo `NaN` explícito y exige que un método solo entre al ranking si produjo pronóstico en el 100% de los orígenes evaluables — nunca promediar MAPEs calculados sobre conjuntos de tamaño distinto. Esto por construcción elimina la mezcla Holt-Winters/Holt y SARIMA/ARIMA(1,1,1) bajo la misma etiqueta.
- **F05**: implementa validación anidada acotada en cómputo (no en cada origen individual del walk-forward completo, que sería demasiado caro para el presupuesto de la §1): usa los **últimos k orígenes** (k=8–12, configurable) como bloque de evaluación para elegir hiperparámetros, y reporta el MASE/MAPE sobre ese bloque como el desempeño del método optimizado — nunca el mínimo del barrido completo. Para SES/Holt conserva el barrido manual acotado (barato); para ARIMA/SARIMA sustitúyelo por `AutoARIMA`/`AutoETS` de `statsforecast`, que selecciona por AICc (verosimilitud, no por MAPE post-hoc) y es a la vez más rápido y estadísticamente más defendible — resuelve F05 y F13 a la vez.
- **F03**: sustituye toda llamada a `_fitted_series` (para el gráfico "pronóstico ajustado sobre histórico" y su MAPE) por `_backtest_with_best` — la función que ya está escrita, es correcta, y nunca se invoca. Renombra la etiqueta de la interfaz a "pronóstico un paso adelante (fuera de muestra)". Si se conserva una vista in-sample como referencia visual, debe ir en gris y sin ninguna métrica asociada, para que no pueda confundirse con desempeño.

Prueba de aceptación: para cada método, `n_preds` debe ser idéntico entre todos los candidatos evaluados en una misma serie (o el método queda excluido, nunca promediado sobre menos puntos). Un test de regresión debe verificar, con una serie sintética conocida, que el pronóstico en el instante t **no** usa `y[t]` en su cómputo (assert de no-fuga programático, no solo visual).

### Fase 4 — Experimento de línea base (F09)
Construye el script `experiments/vs_incumbente.py`: dado el Excel real de la empresa de referencia (o, si no está disponible en este momento, un dataset sintético documentado que reproduzca la estructura descrita en la tesis — ver nota abajo), reconstruye la serie de pronósticos del método incumbente (promedio móvil manual descrito en §3.1 de la tesis) y calcula su MASE/MAPE/ME por walk-forward sobre las mismas fechas que evalúa la herramienta. Produce una tabla única: **incumbente vs. naive vs. herramienta**, sobre datos idénticos. Este es el resultado que hoy falta y el que sostiene la afirmación central del paper — no se declara terminada la fase hasta que este archivo exista y corra de punta a punta.

> Si en el momento de ejecutar este prompt no se cuenta con el Excel real de la empresa de referencia: deja el pipeline completamente parametrizado para aceptarlo (`experiments/vs_incumbente.py --input ruta.xlsx`), documenta explícitamente en el `README.md` que el resultado está pendiente de datos reales, y genera de todos modos la comparación sobre el dataset sintético equivalente para dejar el código verificado end-to-end.

### Fase 5 — Valor operativo: intervalos, inventario, horizonte, lote
- **F20**: añade intervalos de predicción a todos los métodos — `get_forecast().conf_int()` para SARIMAX/ETS; *bootstrap* de residuos del walk-forward para los métodos simples (SES, Holt, promedios, regresión). Grafica la banda y añádela al Excel exportado. A partir de σ del error de walk-forward, implementa `inventory.py`: `SS = z·σ_L`, `ROP = μ_L + SS`, dado un *lead time* y nivel de servicio provistos por el usuario.
- **F21**: acota el horizonte por defecto a `n/3` (nunca más que la historia disponible), añade un piso de no-negatividad al pronóstico, y muestra un aviso explícito cuando el usuario solicita un horizonte mayor.
- **F19**: acepta un Excel con columna `sku` adicional; procesa en lote con `batch.py` (presupuesto de memoria constante — streaming a `parquet`, ver §1), barra de progreso, y `joblib` con `n_jobs` acotado. Salida: tabla maestra SKU · clasificación · método ganador · MASE · MAPE · ME · pronóstico H meses · intervalo · señal de rastreo.

Prueba de aceptación: correr el lote sobre ≥50 series sintéticas de tamaños y estructuras variadas y verificar que la memoria residente del proceso no crece de forma monótona con el número de series (perfilar con `tracemalloc` o `memory_profiler`).

### Fase 6 — Robustez de datos y manejo de errores
- **F15**: acepta meses en formato numérico (1–12) y abreviaturas en inglés; detecta separador de miles europeo vs. decimal y normaliza explícitamente (no adivinar en silencio — preguntar o declarar la convención asumida); detecta filas duplicadas con mensaje propio; ante huecos temporales, **no los elimines en silencio** — repórtalos al usuario y ofrece una decisión explícita (interpolar / mantener como NaN informado / abortar la carga).
- **F16**: elimina `warnings.filterwarnings("ignore")` global; sustituye por `logging` configurable. Cada `except Exception` genérico debe (a) registrar el error con contexto (serie, método, parámetros) y (b) decidir explícitamente si el fallo excluye al método del ranking o aborta la operación — nunca silencio total.

### Fase 7 — Medición de desempeño rigurosa (F25)
Reescribe la medición de tiempos con `time.perf_counter()`, ≥10 repeticiones por tamaño de serie, reporta media±desviación estándar, **incluye explícitamente el Módulo 3** (hoy ausente de la Tabla 3 y responsable de ~60% del tiempo total según la Auditoría B), declara CPU/RAM/SO/versión de Python y de cada librería, y publica el ajuste de complejidad empírica (la Auditoría A midió un exponente ~2.0 con los datos del propio manuscrito — cuadrático, no "progresivo"). Verifica contra el presupuesto de la §1 en el hardware objetivo (o el más cercano disponible) y documenta cualquier brecha.

### Fase 8 — Validación estadística ampliada
1. Re-ejecuta la simulación Monte Carlo de la Auditoría B (1000 réplicas por escenario) sobre la clasificación **ya corregida** (Fase 1) y confirma que las tasas de falso positivo caen del 50–74% actual a ~5% nominal. Este es el test de aceptación definitivo de F10/F11.
2. Construye `experiments/panel_publico.py`: toma un subconjunto **acotado** (100–300 series, no el M4 completo — respeta el presupuesto de RAM) del panel M3 o M4 mensual filtrado a series cortas (≤48 observaciones, el régimen donde este trabajo compite), corre el pipeline completo, y reporta la distribución de MASE por método y por régimen estructural (tendencia/estacional/plana). Añade una prueba de significancia por pares (Diebold-Mariano) o el procedimiento MCB de Koning et al. entre los métodos top.
3. Decide sobre Prophet (F08): dado que Prophet no es el comparador más relevante según la propia Auditoría B (su desempeño mediocre en series cortas es conocido desde M4) y que instalarlo (backend Stan) es costoso en el presupuesto de RAM de la §1, **recomendación**: sustituir la comparación por `AutoARIMA`/`AutoETS`/`AutoTheta` de `statsforecast` — ya integrados por la Fase 3, sin costo adicional de dependencias pesadas — y reescribir la subsección con el protocolo, los datos y la figura reales. Si el usuario insiste en mantener Prophet, documentarlo con el mismo rigor (versión, configuración, datos, código en el repo) antes de reportar un solo número.

### Fase 9 — Reconstrucción del manuscrito académico
- **F07**: elimina todo el andamiaje de plantilla (§0 "How to Use this Template", bloques `ELIMINAR`, secciones de ejemplo con logo de MDPI, bibliografía placeholder); completa `\Title`, autores, afiliaciones, correos reales; genera las figuras reales a partir de los outputs de `experiments/` (nunca reutilizar `flowchart_tool.png` diez veces); resuelve toda referencia cruzada (`Figure NNN`, `Figure X`); **escribe la sección Conclusions**; completa las siete declaraciones obligatorias de MDPI con contenido real (Author Contributions con taxonomía CRediT, Funding, Institutional Review → "Not applicable" si corresponde, Informed Consent, Data Availability apuntando al repositorio con DOI de Zenodo, Acknowledgments con la declaración de GenAI ya bien redactada, Conflicts of Interest).
- **F08**: aplica la decisión de la Fase 8.3.
- **F17**: añade explícitamente, en Limitaciones, que la demanda de la empresa de referencia está dirigida por proyectos de obra y que la variable más predictiva (cartera adjudicada, licencias de construcción) es exógena y conocida — más específico y más honesto que la mención genérica de "variables externas" que hoy tiene el texto.
- **F22 — redacción**: aplica la decisión tomada en la Fase 1 (implementar el filtrado y documentarlo, o retirar la afirmación).
- **F23**: corrige las 11 fórmulas usando la tabla de la Auditoría B como lista de verificación (índice de SES, paréntesis de Holt/Holt-Winters, paréntesis de ARIMA, ecuación de SARIMA ausente, barras de valor absoluto de MAD/MAPE). Verifica cada fórmula corregida contra la implementación real en `forecasting_core/models.py` — deben coincidir exactamente.
- **F24**: corrige años y atribuciones de citas (usa la lista de discrepancias de la Auditoría B como checklist), añade las 11 entradas bibliográficas faltantes o retira las citas que las requieren, migra a formato numérico MDPI con `mdpi.bst` (ya incluido en `Definitions/`). Reformula la narrativa de contribución de "comparación metodológica novedosa" (que un revisor rebatirá citando `statsforecast`, `sktime`, `fable`/`forecast` de R) a **transferencia tecnológica a un régimen de datos escasos** (24–48 observaciones) subrepresentado en la literatura — es el ángulo que ambas auditorías coinciden en que sí es defendible, y que las Fases 4 y 8.2 ya generan evidencia para sostener.

### Fase 10 — Higiene final de repositorio (F18)
`requirements.txt` con versiones fijadas (las mismas con las que corriste las Fases 4–8, para que el próximo auditor no reconcilie nada); `README.md` con instrucciones de ejecución de cada script de `experiments/`; `LICENSE`; semilla fija (`--seed`) propagada a toda generación sintética y a todo barrido con componente aleatorio; `debug=False` en producción; suite `pytest` (ver checklist abajo) corriendo en CI si el repositorio tiene Actions configurables; limpieza de `__MACOSX/`, `__pycache__/` y archivos de plantilla no usados.

---

## 4. Suite de pruebas mínima obligatoria (`tests/`)

No declares el refactor terminado sin que existan y pasen, como mínimo:

1. `test_classification.py` — el predicado de F01 sobre las series de control de ambas auditorías; Monte Carlo de F10/F11 con tasas de falso positivo dentro de tolerancia del nivel nominal.
2. `test_metrics.py` — MASE/MAPE/MAD/MSE/ME correctos contra valores calculados a mano; MAPE con cero no explota.
3. `test_no_leakage.py` — assert programático de que ningún pronóstico en el instante t usa `y[t]` ni información posterior, para cada método del registro.
4. `test_walk_forward_parity.py` — `n_preds` idéntico entre todos los métodos evaluados en una misma serie, o exclusión explícita.
5. `test_model_registry.py` — despacho por clave exacta; un nombre desconocido lanza excepción, nunca cae a otro modelo.
6. `test_data_loading.py` — las siete variantes de entrada de F15 (meses numéricos, meses en inglés, separador de miles, duplicados, huecos) con el comportamiento esperado documentado.
7. `test_batch_memory.py` — memoria acotada en procesamiento por lotes (Fase 5).
8. `test_hyperopt_no_bias.py` — el MASE reportado tras la optimización de hiperparámetros no puede ser igual al mínimo del barrido completo sobre toda la serie (verifica que la Fase 3.F05 realmente usa un bloque separado).

## 5. Entregables finales

Al cerrar todas las fases, el agente debe producir y entregar:
1. Repositorio reestructurado (§Fase 0) con las 25 correcciones aplicadas y trazadas en `CHANGELOG.md` por ID.
2. Resultados de `experiments/` (Fases 4 y 8), incluida la tabla incumbente-vs-naive-vs-herramienta y el reporte de panel público con significancia estadística.
3. `manuscript/template.tex` compilando a PDF sin errores, con todas las declaraciones y figuras reales.
4. Un informe final (`RESUMEN_EJECUCION.md`) que recorra la tabla de la §2 y marque cada uno de los 25 hallazgos como `Corregido y verificado` / `Corregido, pendiente de dato externo` / `Decisión de diseño aplicada (detallar cuál)`.

## 6. Qué NO hacer

- No agregues *machine learning* avanzado ni *deep learning* — el régimen de datos (24–48 observaciones) no lo sostiene, y ambas auditorías coinciden en que la limitación univariante está bien manejada como está. El esfuerzo va a rigor, no a complejidad de modelos.
- No pares el barrido de hiperparámetros en un tope arbitrario sin medir primero si `statsforecast` ya resuelve el problema sin necesidad de barrido manual.
- No optimices prematuramente el 20% del código que no es el cuello de botella (Fase 7 mide primero, optimiza después).
- No dejes ninguna afirmación cuantitativa en el manuscrito sin su script correspondiente en `experiments/` que la reproduzca con una sola invocación.
