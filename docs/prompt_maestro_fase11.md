# PROMPT MAESTRO — Fase 11: Comparación externa (Prophet + LightGBM), más gráficos, manual de usuario e interactividad del repositorio

> Este documento preserva, sin editar, la instrucción que dio origen a la
> Fase 11 (ver `docs/prompt_maestro.md` para las Fases 0-10 / F01-F25 que la
> preceden). Mismo criterio de trazabilidad que ese archivo: el próximo
> auditor debe poder leer exactamente lo que se pidió, no una paráfrasis.

## 0. Origen de este pedido

Instrucción directa de los tutores del proyecto (25/08/2026):

> "Hazle, busca el prophet y otro pronosticador famoso. Dile a Claude que
> haga una comparación con medidas de error de esos vs el nuestro.
> Necesitamos más gráficos. Que ese arreglo no afecte el programa, debe ser
> aparte. [...] Recuerden dejar en el repo esto más interactivo, con el
> manual de funcionamiento del software."

Más un PDF adjunto (`docs/comparacion_herramientas.pdf`) con 10 corridas
manuales de "Herramienta vs. Prophet" sobre series de 24 a 180 meses. Ese PDF
NO es reutilizable como evidencia: mezcla dos protocolos de evaluación
distintos (ver hallazgo F26). Se usó solo como inspiración de formato visual
(gráfico dual histórico+pronóstico lado a lado, tabla de métricas debajo).

## 1. Restricción de aislamiento — no negociable

Esta fase vive en un espacio aparte del núcleo ya validado por la suite
`pytest` existente:

- Nuevo paquete hermano `codigo/external_baselines/` (NO dentro de
  `forecasting_core/`). Imports de `prophet` y `mlforecast`/`lightgbm`
  perezosos (dentro de la función, no a nivel de módulo) para que
  `import forecasting_core` y `python codigo/app.py` sigan funcionando sin
  esas librerías instaladas.
- Nuevo archivo `requirements-external.txt` (prophet, cmdstanpy, mlforecast,
  lightgbm, con versiones fijadas) — NO tocar `requirements.txt`, que debe
  seguir siendo la instalación mínima para correr el núcleo y la app.
- La suite existente de pruebas (`pytest` desde la raíz) debe seguir pasando
  exactamente igual al terminar esta fase, sin ninguna dependencia nueva.
  Verificarlo y reportarlo en el resumen final.
- Mismo presupuesto de hardware del prompt original (Ryzen 5 7000, 8 GB RAM,
  ~5-6 GB útiles) para la SESIÓN INTERACTIVA (`app.py`). El experimento de
  comparación (Fase 11.2) puede tardar más porque corre offline/batch, pero
  se debe documentar su tiempo total igualmente.

## 2. Nuevos hallazgos y requerimientos (continúan la numeración F01-F25)

| ID | Título | Origen |
|---|---|---|
| F26 | `comparacion_herramientas.pdf` mezcla dos protocolos de evaluación distintos (MAPE walk-forward interno de la Herramienta vs. métricas de Prophet sobre un holdout aparte, con estacionalidad anual de Prophet activada sobre 24 meses de historia — produce MAPE=353.98% y pronósticos negativos en el set n=24). No es una comparación válida ni reutilizable en el paper tal como está. | Hallazgo propio, a partir del PDF adjunto |
| F27 | Falta un módulo de benchmarks externos (Prophet + LightGBM) evaluado bajo el MISMO arnés walk-forward / `honest_outer_estimate` ya validado en la Fase 3, aislado del núcleo para no romper pruebas ni el presupuesto de RAM interactivo. | Tutoría 25/08/2026 |
| F28 | Faltan tipos de gráfico adicionales para comunicar desempeño: distribución de error por método, precisión vs. longitud de serie, panel de pronósticos por régimen estructural. | Tutoría 25/08/2026 ("necesitamos más gráficos") |
| F29 | Falta manual de funcionamiento del software y mayor interactividad del repositorio (modo demo con datos de ejemplo, pestaña de comparación externa en la interfaz). | Tutoría 25/08/2026 |
| F30 | Título del artículo no convence a los autores. | Tutoría 25/08/2026 — FUERA DE ALCANCE de este prompt; se resuelve en la Fase 12, reescritura del paper, después de revisar los resultados de esta fase. |

## 3. Decisión de modelos (reabre F08 bajo instrucción explícita)

`decision_prophet.md` retiró Prophet por (a) desempeño mediocre conocido en
series cortas y (b) costo de dependencia (backend Stan) frente al
presupuesto de RAM de la sesión interactiva. Ambos argumentos siguen siendo
válidos para la APP INTERACTIVA — por eso Prophet queda aislado y opcional
(§1), nunca como dependencia obligatoria. Pero los tutores piden
explícitamente reincorporarlo para la comparación empírica del paper, así
que:

- **Modelo 1: Prophet** (paquete `prophet`, backend cmdstanpy). Configurado
  respetando sus propios supuestos documentados: `yearly_seasonality` solo
  se activa si hay ≥ 24 observaciones (2 ciclos anuales completos); con
  menos, se desactiva explícitamente y se documenta la decisión en el
  docstring — replicar el error del PDF adjunto (activarla siempre) sería
  repetir F26 a propósito.
- **Modelo 2: LightGBM** como modelo global vía `mlforecast` (familia
  Nixtla, consistente con `statsforecast` ya usado; ganador histórico de
  M5). Usado con features de lags/rolling estándar de mlforecast, sin
  ajuste fino agresivo (esto es una línea base, no una tesis aparte sobre
  tuning de gradient boosting).

Si por alguna razón de instalación ninguno de los dos fuera viable en el
entorno de ejecución, documentar el bloqueo exacto en
`codigo/experimentos/decision_prophet.md` (actualizarlo, no borrarlo) y
continuar con el que sí funcione — no dejar la fase completa bloqueada por
un solo paquete.

## 4. Fase 11 — subfases

### 11.1 Módulo de benchmarks externos aislado

`codigo/external_baselines/`:
- `prophet_adapter.py`: `fit_predict_prophet(y, h, freq="MS") -> np.ndarray`,
  import perezoso, no-negatividad aplicada de forma consistente con
  `ModelSpec.forecast(..., clip_non_negative=True)` del núcleo (mismo
  criterio, sin importar el núcleo para aplicarlo).
- `lightgbm_adapter.py`: función equivalente vía `mlforecast`.
- `adapters.py`: envuelve ambas funciones en un objeto con la MISMA forma
  que `forecasting_core.models.ModelSpec` (fit/predict/forecast), para que
  puedan pasarse a `forecasting_core.validation.walk_forward` y a
  `forecasting_core.optimize.honest_outer_estimate` SIN modificar esos
  módulos — patrón de composición, no de edición. Si `honest_outer_estimate`
  exige un `ModelSpec` real y no un duck-type, añadir el hook aditivo mínimo
  indispensable en `optimize.py` (una función que acepte una lista de specs
  externos ya construidos) y documentar esa única edición en el CHANGELOG
  bajo F27 — es la única excepción permitida a "no tocar forecasting_core".
- Tests nuevos en `codigo/tests/test_external_baselines.py`, con
  `pytest.importorskip("prophet")` / `pytest.importorskip("mlforecast")` al
  inicio de cada clase de test, para que la suite principal nunca falle por
  falta de estas dependencias opcionales.

### 11.2 Experimento honesto de comparación externa

`codigo/experimentos/comparativa_externa.py`, mismo patrón que
`vs_incumbente.py` / `panel_publico.py` (semilla fija, un solo comando,
escribe CSV + log versionados en `resultados/`):

- Panel sintético con longitudes n ≈ 24, 36, 48, 60, 72, 84, 96, 120, 150,
  180 — mismo barrido del PDF adjunto — con al menos estos regímenes
  estructurales por longitud: plano, tendencia, estacional,
  tendencia+estacional, y corta/errática (el caso n=24 que rompió a
  Prophet en el PDF).
- Evalúa Herramienta (ganador del pipeline corregido) vs. Prophet vs.
  LightGBM, LOS TRES bajo el mismo protocolo de tres bloques (tune/eval/
  outer) de `honest_outer_estimate` — nunca la métrica in-sample de un lado
  contra el holdout del otro (ver F26).
- Métricas: MASE (primaria), MAPE, MAD, MSE, ME — mismo set que
  `forecasting_core.metrics`.
- Salida: `resultados/comparativa_externa.csv` +
  `resultados/logs/comparativa_externa.log`.
- Si el archivo real de Tuboplex sigue sin estar disponible, usar el mismo
  panel sintético de `vs_incumbente.py` (acepta `--input` igual que ese
  script, por consistencia).

### 11.3 Gráficos nuevos (F28)

Extender `codigo/experimentos/make_figures.py` (o crear
`make_figures_comparativa.py` si se prefiere mantenerlo separado, respetando
el mismo `OUT = manuscritos/articulo_mdpi/figures/` y estilo matplotlib):
- Boxplot de MASE por método (Herramienta / Prophet / LightGBM / naive).
- Dispersión precisión (MASE) vs. longitud de serie, un color por método.
- Panel de pequeños múltiplos: para 3-4 regímenes representativos, histórico
  + pronóstico de los tres métodos superpuestos.
- Actualizar el mapeo script→figura→sección en el README.

### 11.4 Superficie opcional en la interfaz (F27, F29 parcial)

En `codigo/app.py`, agregar una pestaña/sección nueva "Comparación externa
(Prophet / LightGBM)":
- Import guard: si `prophet`/`mlforecast` no están instalados, la pestaña se
  muestra deshabilitada con un mensaje, y el resto de la app funciona
  exactamente igual que hoy.
- Si están instalados: sobre la serie ya cargada en el Módulo 1, correr los
  adaptadores de 11.1 y mostrar el mismo panel dual definido en 11.3.
- Ninguna de las cuatro pestañas/módulos existentes cambia de
  comportamiento, orden, ni callbacks.

### 11.5 Manual de usuario e interactividad del repo (F29)

- Nuevo `docs/MANUAL_USUARIO.md` (español): instalación, cómo correr
  `python codigo/app.py`, recorrido de los cuatro módulos existentes más el
  nuevo módulo de comparación externa, cómo exportar resultados, cómo usar
  `batch_cli.py` para multi-SKU.
- Modo demo: botón "Cargar datos de ejemplo" en el Módulo 1 que carga una
  serie sintética embebida — un solo callback adicional, no toca la lógica
  de carga real.
- Actualizar el README: enlazar el manual nuevo, documentar cómo instalar
  `requirements-external.txt`, agregar `comparativa_externa.py` a la tabla
  de "Reproducir los resultados del manuscrito".

### 11.6 No regresión

- Correr `pytest` completo desde la raíz y confirmar las pruebas originales
  sin cambios, más las nuevas de 11.1 (que se saltan automáticamente si
  prophet/mlforecast no están instalados).
- Agregar una entrada nueva a `CHANGELOG.md` bajo un encabezado "## Fase 11
  — Comparación externa" citando F26-F29 con el mismo formato que las
  entradas F01-F25 existentes.

## 5. Prueba de aceptación de la fase

No declarar la fase terminada sin:
a) `pytest` → pruebas originales en verde, sin excepción.
b) `resultados/comparativa_externa.csv` existe y fue generado con un solo
   comando reproducible con semilla fija.
c) Las tres figuras nuevas de 11.3 existen en
   `manuscritos/articulo_mdpi/figures/`.
d) `python codigo/app.py` arranca y funciona igual que antes de esta fase
   SIN `requirements-external.txt` instalado.
e) `docs/MANUAL_USUARIO.md` existe y cubre los cinco módulos.

## 6. Qué NO hacer en esta fase

- No editar `manuscritos/articulo_mdpi/template.tex` ni la versión en
  español del paper — el título (F30) y la reescritura con estos resultados
  nuevos se hacen en la Fase 12, después de revisar el CSV y las figuras de
  esta fase.
- No editar `forecasting_core/{data,classification,metrics,validation,
  intervals,inventory,batch}.py`. Solo `optimize.py` puede recibir el hook
  aditivo mínimo descrito en 11.1, y debe quedar documentado como tal.
- No agregar Prophet ni LightGBM a `requirements.txt` ni a `MODEL_REGISTRY`
  del núcleo — viven en `external_baselines/` y `requirements-external.txt`.
- No hacer tuning agresivo de LightGBM ni Prophet: son líneas base para
  comparar, no un nuevo modelo estrella del paper.

## 7. Entregables al terminar

1. Rutas de todos los archivos nuevos/editados.
2. Resultado íntegro de `pytest` (conteo antes/después).
3. Tabla resumen de `resultados/comparativa_externa.csv` (MASE por método y
   por régimen).
4. Lista de las figuras nuevas generadas, con su ruta.
5. Confirmación explícita del punto 5.d (app funciona sin las deps
   opcionales instaladas).

---

## Nota de ejecución (post-hoc)

Ejecutada el 25-26/08/2026. Resultado real (no anticipado por el prompt):
el hook aditivo de `optimize.py` previsto en 11.1 resultó **innecesario** —
`PROPHET_SPEC`/`LIGHTGBM_SPEC` son instancias reales de `ModelSpec` que se
pasan directamente a `walk_forward`/`backtest_one_step` sin ninguna
modificación al núcleo, siguiendo el mismo patrón que ya usa
`vs_incumbente.py` para el método incumbente. `forecasting_core/optimize.py`
no se tocó. Ver `codigo/external_baselines/adapters.py` (docstring) y
`CHANGELOG.md` (sección "Fase 11") para el detalle completo, y
`RESUMEN_EJECUCION.md`/el resumen de cierre de esta fase para los resultados
cuantitativos (`resultados/comparativa_externa.csv`, 50/50 series
evaluadas).
