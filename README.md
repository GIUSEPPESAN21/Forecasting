# Motor de Pronósticos — Tuboplex

Herramienta de pronóstico de demanda con clasificación estructural, validación
honesta (bloque de ajuste de hiperparámetros separado del bloque de
evaluación), intervalos de predicción y política de inventario derivada.

Este repositorio es un refactor completo de una versión previa que contenía
defectos estadísticos que invalidaban sus propios resultados publicados (ver
`CHANGELOG.md` y `RESUMEN_EJECUCION.md` para el detalle). El núcleo de
pronóstico (`forecasting_core/`) no depende de Dash ni de Plotly y es
testeable de forma independiente.

## Estructura

```
forecasting_core/    Núcleo de pronóstico (sin dependencias de interfaz)
  data.py               Carga y validación de series (F15)
  classification.py     Tendencia / estacionalidad / estacionariedad (F01, F10, F11)
  models.py             Registro explícito de modelos (F04)
  metrics.py             MASE primaria, MAPE seguro ante ceros, ME (F06, F12)
  validation.py         Walk-forward de una sola pasada, con paridad (F02, F03, F14)
  optimize.py            Tuning anidado y ganador sin sesgo de selección (F05, F13)
  intervals.py            Intervalos de predicción empíricos (F20)
  inventory.py            Stock de seguridad y punto de reorden (F20)
  batch.py                Procesamiento multi-SKU con memoria acotada (F19)
app.py                  Interfaz Dash (capa delgada sobre forecasting_core)
batch_cli.py             CLI de procesamiento por lotes multi-SKU (F19)
tests/                   Suite pytest (ver más abajo)
experiments/             Scripts de validación empírica y reproducibilidad
manuscript/              Manuscrito LaTeX (MDPI, journal Forecasting)
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Probado con Python 3.11.9. Las versiones de librería están fijadas en
`requirements.txt` a las usadas para producir los resultados de
`experiments/output/`.

## Ejecutar la aplicación

```bash
python app.py
```

Por defecto corre en modo producción (`debug=False`). Para depurar:
`FORECASTING_DEBUG=true python app.py`. El nivel de log se controla con
`FORECASTING_LOG_LEVEL` (por defecto `WARNING`).

## Ejecutar las pruebas

```bash
pytest                       # suite completa, excluye pruebas lentas por defecto en CI
pytest -m slow               # incluye el perfilado de memoria del lote
```

La suite cubre: potencia y tamaño de la clasificación estructural, métricas
(incluyendo MASE y el caso de demanda cero), ausencia de fuga temporal
(verificada programáticamente, no solo por inspección), paridad del
walk-forward, despacho exacto del registro de modelos, ausencia de sesgo de
selección de hiperparámetros, carga robusta de datos, y memoria acotada del
procesamiento por lotes.

## Reproducir los resultados del manuscrito

Cada cifra cuantitativa de `manuscript/template.tex` proviene de uno de estos
scripts, ejecutable con un solo comando y semilla fija:

```bash
# Validación Monte Carlo de la clasificación estructural (Sección 1 del refactor)
python experiments/montecarlo_clasificacion.py --reps 1000 --sizes 24 36 48 120

# Caso ilustrativo end-to-end (Sección 3.2 del manuscrito)
python experiments/caso_ilustrativo.py

# Herramienta vs. método incumbente de Tuboplex (Sección 3.3)
python experiments/vs_incumbente.py --synthetic --n-series 30 --seed 20260824
# Con el archivo real de la empresa:
python experiments/vs_incumbente.py --input ruta/al/archivo.xlsx

# Validación sobre panel público M3-Monthly truncado (Sección 3.4)
python experiments/panel_publico.py --n-series 150 --max-len 48 --seed 20260824

# Tiempos computacionales (Sección 3.5)
python experiments/benchmark_tiempos.py --reps 5 --sizes 24 48 72 96 120

# Figuras del manuscrito
python experiments/make_figures.py
```

Los resultados quedan en `experiments/output/` (no versionado; se regenera).

## Procesamiento por lotes (multi-SKU)

Desde la línea de comandos, sin escribir Python:

```bash
python batch_cli.py productos.xlsx salida/ --horizon 12 --lead-time 3 --service-level 0.95
```

O desde código:

```python
from forecasting_core.batch import BatchConfig, run_batch
import pandas as pd

panel = pd.read_excel("productos.xlsx")  # columnas: sku, year, month, demand
run_batch(panel, "salida/", BatchConfig(horizon=12, lead_time=3, service_level=0.95))
```

Memoria acotada: los resultados se vuelcan a disco incrementalmente, no se
acumula el portafolio completo en memoria (ver `tests/test_batch_memory.py`).

## Limitaciones conocidas

- Modelos univariados: no incorpora variables exógenas (p. ej. cartera
  adjudicada de proyectos de construcción).
- Los datos reales de Tuboplex no están incluidos (confidencialidad
  comercial); `experiments/vs_incumbente.py` acepta el archivo real como
  reemplazo directo del panel sintético.
- Con menos de ~36 observaciones mensuales la estacionalidad no es
  identificable de forma confiable (ver `forecasting_core/classification.py`).

## Licencia

MIT — ver `LICENSE`.
