# Decisión: retirar la comparación con Prophet (F08, Fase 8.3)

## Que habia en el manuscrito

`template.tex`, subseccion "Comparison with Prophet": cuatro parrafos que
afirman un patron de dominancia diferenciado entre la herramienta y Prophet
segun el tamano de muestra ("para series de 24 observaciones la herramienta
supera sustancialmente a Prophet"), sin figura, sin tabla, sin un solo numero,
sin codigo de Prophet en ningun archivo del proyecto, sin version declarada, sin
descripcion de las series usadas ni de cuantas eran, y sin prueba estadistica.
Es, literalmente, un resultado inventado.

## Por que no se reconstruye con Prophet real

1. **No es el comparador mas relevante.** El desempeno mediocre de Prophet en
   series cortas y de baja frecuencia es un hallazgo conocido desde la
   competencia M4 (Makridakis et al. 2020): Prophet quedo por debajo de
   metodos estadisticos simples y de `ets()`/`auto.arima()` en la mayoria de
   los regimenes de esa competencia, particularmente con historias cortas.
   Ganarle a Prophet con 24 observaciones no demuestra que la herramienta sea
   competitiva; demuestra que Prophet esta mal elegido como referencia.

2. **Costo de dependencias frente al presupuesto de la Fase 1.** Prophet usa
   un backend Stan (PyStan/CmdStanPy) que compila un modelo bayesiano por
   ajuste. En un equipo de 8 GB de RAM total (~5-6 GB utiles), anadir esa
   dependencia para un comparador que ya se sabe subordinado no se justifica.

3. **`statsforecast` ya esta integrado (Fase 3) y no cuesta nada adicional.**
   `AutoARIMA`, `AutoETS` y `AutoTheta` seleccionan por AICc, son JIT (Numba) y
   ya forman parte de `MODEL_REGISTRY` porque resuelven F05/F13 (el barrido
   manual de hiperparametros de ARIMA/SARIMA). Compararse contra ellos no
   agrega dependencias: mide contra el estado del arte actual sin gastar
   presupuesto de memoria extra.

## Que reemplaza a la seccion

Dos comparaciones, ambas ya ejecutables con el codigo de este refactor:

- **`experiments/vs_incumbente.py`**: herramienta vs. metodo incumbente de
  Tuboplex (promedio movil k=3) vs. naive. Es el resultado que sostiene la
  afirmacion central del paper (Fase 4 / F09).
- **`experiments/panel_publico.py`**: herramienta vs. naive/seasonal_naive
  sobre un panel de M3 mensual truncado a series cortas (Fase 8.2), con
  significancia estadistica (Wilcoxon signed-rank sobre MASE por serie). Al
  usar el mismo `MODEL_REGISTRY`, el desempeno de `auto_arima`/`auto_ets`/
  `auto_theta` ya queda documentado en la distribucion de metodos ganadores por
  regimen que ese script reporta -es la comparacion contra el estado del arte
  moderno que Prophet pretendia ofrecer, sin el costo de la dependencia.

## Redaccion sugerida para el manuscrito (Seccion 3, Fase 9)

Sustituir la subseccion "Comparison with Prophet" por una subseccion
"Comparison against automated statistical baselines", que reporte los
resultados de `panel_publico.py`: la fraccion de series donde la herramienta
supera al naive estacional por regimen estructural, y la frecuencia con la que
`auto_arima`/`auto_ets`/`auto_theta` resultan ganadores frente a los metodos
clasicos (SES, Holt, Holt-Winters, regresion) cuando ambos compiten bajo el
mismo filtro estructural. Esto reemplaza una afirmacion sin evidencia por un
resultado reproducible con un solo comando.

## Si en el futuro se quiere reincorporar Prophet

Es tecnicamente posible (`pip install prophet`) y el patron de integracion es
identico al de `statsforecast` en `forecasting_core/models.py`: una funcion
`_f_prophet(y, p, h, m)` y una entrada en `MODEL_REGISTRY`. Si se hace, debe
documentarse version, configuracion (`seasonality_mode`, `changepoint_prior_scale`,
etc.), las series usadas y el codigo debe vivir en el repositorio -exactamente
el estandar que esta decision aplica a todo lo demas.
