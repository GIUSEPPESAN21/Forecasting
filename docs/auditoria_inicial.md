# Auditoría tesisVF.py + TESIS!.docx + template.tex — hallazgos verificados

Informe completo (artifact): https://claude.ai/code/artifact/31075df1-5b44-4754-9a86-d981d735cb27
Commit auditado: GIUSEPPESAN21/Forecasting @ 63bf54e

## Veredicto
NO listo para someter a MDPI Forecasting. Reject / Major revision.
Puntuación: técnica 3.5/10 · paper 2.5/10 · ingeniería industrial 4.5/10.

## Defectos críticos (verificados ejecutando el código)

- **C-01 — `es_muy_lineal()` es insatisfacible** (líneas 259-279). Exige R²≥0.90 Y |ACF12|<0.10,
  condiciones mutuamente excluyentes. Devolvió True 0 veces en 3.000 series simuladas.
  Consecuencia: Promedio Simple/Móvil/Ponderado se excluyen SIEMPRE del ranking (líneas 439, 833).
  **El resultado del §3.2.5 del .docx ("serie estacionaria → gana Promedio Simple") es irreproducible.**

- **C-02 — Comparación no homogénea** (líneas 340-371). Con n=24: n_preds = 18 para los promedios,
  14 para SES/Holt, y **4 para Holt-Winters**, y esos 4 puntos ejecutan `f_holt`, no Holt-Winters
  (statsmodels exige 2·m=24 obs para el modelo estacional; `heuristic` exige ≥10).
  Contradice "cada método se ejecuta bajo las mismas condiciones".

- **C-03 — Fuga in-sample en el "MAPE histórico ajustado"** (líneas 1004-1012, 1146-1152).
  `rolling(k).mean()` incluye el propio y[t]; en el ponderado con peso 0.5.
  La función correcta `_backtest_with_best` (1087-1117) existe y **nunca se llama**.

- **C-04 — `_make_predictor` cae al fallback naive** (602-604) para los tres promedios.
  El Módulo 3 reporta el MAPE de un paseo aleatorio como "MAPE optimizado" (7.79% vs 5.30% real).

- **C-05 — Sesgo de selección**: los hiperparámetros se eligen sobre el mismo walk-forward cuyo MAPE
  se reporta (700-765). Además, en el ranking SES/Holt/HW compiten con α=β=γ=0.1 fijo contra métodos
  sin parámetros → comparación injusta.

- **C-06 — ME (Error Medio) declarado en ambos manuscritos, no implementado** (249-254).
  Es la métrica de sesgo, la más relevante para la decisión de inventario.

- **C-07 — template.tex es una plantilla sin terminar y no compila**: §0 "How to Use this Template",
  bloques ELIMINAR, \Title{Title}, afiliaciones placeholder, **Conclusiones vacías**, 10 figuras
  apuntando a `flowchart_tool.png` (archivo inexistente), referencias "Figure NNN" sin resolver,
  las 7 declaraciones obligatorias MDPI con texto boilerplate, bibliografía de ejemplo.

- **C-08 — "Comparison with Prophet"** reporta resultados cuantitativos sin figura, sin datos,
  sin protocolo. Prophet no aparece en el código.

- **C-09 — La afirmación central ("mejora significativamente la precisión") no está medida**:
  el error del método incumbente nunca se calculó (el propio §5 lo admite). Sin benchmark ingenuo
  ni MASE, un MAPE de 22.63% no es interpretable.

## Mayores
M-01 detector de estacionalidad: 100% falsos positivos sobre tendencia pura (|ACF12| medio 0.84-0.98);
umbral 0.30 ≈ nivel de ruido con n=24. M-02 ADF con `regression="c"` sobre series con tendencia;
p-valor de tendencia inválido bajo autocorrelación. M-03 MAPE con eps=1e-8 explota a 1.25e10% con un
solo cero. M-04 topes de cómputo declarados no corresponden (HW acotado a 300 pero usa ≤250; Holt 361
combos sin tope; SARIMA 144 combos = 2.5 min con n=48). M-05 el walk-forward se ejecuta 3 veces por
sesión (líneas 433, 468, 828). M-06 carga falla con meses numéricos, meses en inglés, separador de
miles y filas duplicadas; los huecos se colapsan y desfasan el eje temporal. M-07 16 `except Exception`
silenciosos. M-11 tiempos medidos con cronómetro manual, n=1; los datos del propio manuscrito muestran
crecimiento cuadrático (exponente 2.02-2.04), no "escalable". M-12 repo sin README/requirements/
LICENSE/datos/tests. M-14 sin procesamiento multi-SKU (`multiple=False`). M-15 no entrega σ del error
ni intervalos → no cierra hacia la decisión de inventario.

## Plan (orden no negociable)
1. Semana 1-2: reparar código (tests → C-01 → C-04 → C-06 → C-03 → C-02 → M-01/M-02 → M-03).
2. Semana 2-4: rehacer TODOS los experimentos + línea base (incumbente vs ingenuo vs herramienta)
   + panel público M3/M4 con MASE y prueba de significancia + timing instrumentado.
3. Semana 4-6: reescribir el manuscrito sobre el .tex (abandonar el .docx como fuente).
4. Semana 6: completar el repositorio antes del envío.

Opcional alto retorno: quinto módulo de inventario (punto de reorden + stock de seguridad desde los
residuos del walk-forward).