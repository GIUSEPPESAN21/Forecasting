# Motor de Pronósticos — Tuboplex

Herramienta de pronóstico de demanda con clasificación estructural, validación
honesta (bloque de ajuste de hiperparámetros separado del bloque de
evaluación), intervalos de predicción y política de inventario derivada.

Este repositorio es un refactor completo de una versión previa que contenía
defectos estadísticos que invalidaban sus propios resultados publicados (ver
`CHANGELOG.md` y `RESUMEN_EJECUCION.md` para el detalle). El núcleo de
pronóstico (`codigo/forecasting_core/`) no depende de Dash ni de Plotly y es
testeable de forma independiente.

## Estructura del repositorio

```
codigo/                       Todo el software
  forecasting_core/             Núcleo de pronóstico (sin dependencias de interfaz)
    data.py                       Carga y validación de series (F15)
    classification.py             Tendencia / estacionalidad / estacionariedad (F01, F10, F11)
    models.py                     Registro explícito de modelos (F04)
    metrics.py                     MASE primaria, MAPE seguro ante ceros, ME (F06, F12)
    validation.py                 Walk-forward de una sola pasada, con paridad (F02, F03, F14)
    optimize.py                    Tuning anidado y ganador sin sesgo de selección (F05, F13)
    intervals.py                    Intervalos de predicción empíricos (F20)
    inventory.py                    Stock de seguridad y punto de reorden (F20)
    batch.py                        Procesamiento multi-SKU con memoria acotada (F19)
  app.py                          Interfaz Dash (capa delgada sobre forecasting_core)
  batch_cli.py                     CLI de procesamiento por lotes multi-SKU (F19)
  tests/                           Suite pytest (ver más abajo)
  experimentos/                    Scripts de validación empírica y reproducibilidad

resultados/                    Evidencia versionada: CSV y logs de cada corrida citada
                                en el manuscrito (ver "Resultados versionados" abajo)

manuscritos/
  articulo_mdpi/                  Manuscrito LaTeX (MDPI, journal Forecasting)
  tesis_original/                  Documento de tesis original (Universidad de los Andes)

docs/                           Documentación de proceso: auditoría inicial y prompt maestro
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Probado con Python 3.11.9. Las versiones de librería están fijadas en
`requirements.txt` a las usadas para producir los resultados de `resultados/`.

## Ejecutar la aplicación

```bash
python codigo/app.py
```

Por defecto corre en modo producción (`debug=False`). Para depurar:
`FORECASTING_DEBUG=true python codigo/app.py`. El nivel de log se controla con
`FORECASTING_LOG_LEVEL` (por defecto `WARNING`).

## Ejecutar las pruebas

Desde la raíz del repositorio (`pytest.ini` ya apunta a `codigo/tests`):

```bash
pytest                       # suite completa, excluye pruebas lentas por defecto
pytest -m slow               # incluye el perfilado de memoria del lote
```

La suite cubre: potencia y tamaño de la clasificación estructural, métricas
(incluyendo MASE y el caso de demanda cero), ausencia de fuga temporal
(verificada programáticamente, no solo por inspección), paridad del
walk-forward, despacho exacto del registro de modelos, ausencia de sesgo de
selección de hiperparámetros, carga robusta de datos, y memoria acotada del
procesamiento por lotes.

## Reproducir los resultados del manuscrito

Cada cifra cuantitativa de `manuscritos/articulo_mdpi/template.tex` proviene de
uno de estos scripts, ejecutable con un solo comando y semilla fija. Todos
escriben directamente en `resultados/`:

```bash
# Validación Monte Carlo de la clasificación estructural (Sección 2.3 del manuscrito)
python codigo/experimentos/montecarlo_clasificacion.py --reps 1000 --sizes 24 36 48 120

# Caso ilustrativo end-to-end (Sección 3.2 del manuscrito)
python codigo/experimentos/caso_ilustrativo.py

# Herramienta vs. método incumbente de Tuboplex (Sección 3.3)
python codigo/experimentos/vs_incumbente.py --synthetic --n-series 40 --seed 20260824
# Con el archivo real de la empresa:
python codigo/experimentos/vs_incumbente.py --input ruta/al/archivo.xlsx

# Validación sobre panel público M3-Monthly truncado (Sección 3.4)
python codigo/experimentos/panel_publico.py --n-series 150 --max-len 48 --seed 20260824

# Tiempos computacionales (Sección 3.5)
python codigo/experimentos/benchmark_tiempos.py --reps 5 --sizes 24 48 72 96 120

# Figuras del manuscrito (escriben en manuscritos/articulo_mdpi/figures/)
python codigo/experimentos/make_figures.py
```

## Resultados versionados

A diferencia de un `.gitignore` que oculta toda salida de experimentos, los
CSV y logs finales usados en el manuscrito **sí están versionados** en
`resultados/`, como evidencia trazable de cada cifra publicada:

```
resultados/
  montecarlo_clasificacion.csv     1000 réplicas x 4 tamaños de serie
  caso_ilustrativo_ranking.csv       ranking del caso ilustrativo (Sección 3.2)
  caso_ilustrativo_pronostico.csv    pronóstico + intervalo del caso ilustrativo
  vs_incumbente.csv                 40 series, herramienta vs. incumbente vs. naive
  panel_publico.csv                 150 series M3-Monthly, protocolo de bloque externo
  benchmark_tiempos.csv             tiempos de pipeline por tamaño de serie
  logs/                             transcripciones de consola de cada corrida
```

Solo se excluye del control de versiones `codigo/experimentos/m3cache/`: el
caché del dataset público M3 descargado automáticamente por
`panel_publico.py` (reproducible desde su fuente original, no autorado).

## Procesamiento por lotes (multi-SKU)

Desde la línea de comandos, sin escribir Python:

```bash
python codigo/batch_cli.py productos.xlsx salida/ --horizon 12 --lead-time 3 --service-level 0.95
```

O desde código:

```python
from forecasting_core.batch import BatchConfig, run_batch
import pandas as pd

panel = pd.read_excel("productos.xlsx")  # columnas: sku, year, month, demand
run_batch(panel, "salida/", BatchConfig(horizon=12, lead_time=3, service_level=0.95))
```

Memoria acotada: los resultados se vuelcan a disco incrementalmente, no se
acumula el portafolio completo en memoria (ver `codigo/tests/test_batch_memory.py`).

## Limitaciones conocidas

- Modelos univariados: no incorpora variables exógenas (p. ej. cartera
  adjudicada de proyectos de construcción).
- Los datos reales de Tuboplex no están incluidos (confidencialidad
  comercial); `codigo/experimentos/vs_incumbente.py` acepta el archivo real
  como reemplazo directo del panel sintético.
- Con menos de ~36 observaciones mensuales la estacionalidad no es
  identificable de forma confiable (ver `codigo/forecasting_core/classification.py`).

## Licencia

MIT — ver `LICENSE`.
