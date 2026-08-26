# Manual de usuario — Motor de Pronósticos Tuboplex

[⬅ Volver al README](../README.md) · [Índice del manual](#índice)

Este manual explica cómo instalar y usar la herramienta **sin necesidad de
leer el código**. Si busca el detalle técnico de qué se corrigió y por qué,
vea [`CHANGELOG.md`](../CHANGELOG.md) y [`RESUMEN_EJECUCION.md`](../RESUMEN_EJECUCION.md);
este documento es solo para usar la aplicación.

## Índice

1. [Instalación](#1-instalación)
2. [Arrancar la aplicación](#2-arrancar-la-aplicación)
3. [Recorrido de los cinco módulos](#3-recorrido-de-los-cinco-módulos)
   - [Módulo 1 — Carga y validación de datos](#módulo-1--carga-y-validación-de-datos)
   - [Módulo 2 — Clasificación y evaluación de métodos](#módulo-2--clasificación-y-evaluación-de-métodos)
   - [Módulo 3 — Pronóstico con intervalos](#módulo-3--pronóstico-con-intervalos)
   - [Módulo 4 — Política de inventario](#módulo-4--política-de-inventario)
   - [Módulo 5 — Comparación externa (Prophet / LightGBM)](#módulo-5--comparación-externa-prophet--lightgbm)
4. [Exportar resultados](#4-exportar-resultados)
5. [Procesar muchos productos a la vez (`batch_cli.py`)](#5-procesar-muchos-productos-a-la-vez-batch_clipy)
6. [Preguntas frecuentes / problemas comunes](#6-preguntas-frecuentes--problemas-comunes)

---

## 1. Instalación

Necesita Python 3.11 instalado. Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Con esto ya puede usar los Módulos 1 a 4 de la aplicación. El **Módulo 5**
(comparación con Prophet y LightGBM) es **opcional** y requiere un paso extra
de instalación — ver la sección del Módulo 5 más abajo.

## 2. Arrancar la aplicación

Desde la raíz del repositorio:

```bash
python codigo/app.py
```

La consola mostrará una dirección como `http://127.0.0.1:8050`. Ábrala en su
navegador (Chrome, Edge o Firefox). Para cerrar la aplicación, vuelva a la
consola y presione `Ctrl+C`.

Si el puerto 8050 ya está en uso, puede indicar otro:

```bash
PORT=8060 python codigo/app.py          # Linux/Mac
set PORT=8060 && python codigo/app.py   # Windows (cmd)
$env:PORT=8060; python codigo/app.py    # Windows (PowerShell)
```

## 3. Recorrido de los cinco módulos

La aplicación es **una sola página**: los módulos aparecen uno debajo del
otro, en orden, y cada uno se activa automáticamente cuando el anterior
produce un resultado. No hay que guardar ni confirmar nada entre módulos.

### Módulo 1 — Carga y validación de datos

Arriba de todo hay una zona para arrastrar o seleccionar un archivo Excel
(`.xlsx`) o CSV con **tres columnas obligatorias**:

| Columna  | Contenido                                            | Ejemplos válidos                    |
|----------|-------------------------------------------------------|--------------------------------------|
| `year`   | Año                                                    | `2023`, `2023.0`                     |
| `month`  | Mes                                                    | `1`-`12`, `enero`, `January`, `ene`, `jan` |
| `demand` | Demanda del mes (un número)                            | `1234.5`, `1.234,5` (formato europeo se detecta solo) |

Si tiene varios productos en el mismo archivo, agregue una cuarta columna
`sku` con el nombre o código de cada producto (la app analiza un producto por
vez; para varios productos de una sola corrida, use el
[procesamiento por lotes](#5-procesar-muchos-productos-a-la-vez-batch_clipy)).

**¿No tiene un Excel a mano?** Use el botón **"Cargar datos de ejemplo"**,
justo debajo de la zona de arrastre: carga una serie sintética de 48 meses
(con tendencia y estacionalidad) para que pueda probar toda la herramienta de
inmediato. No reemplaza sus datos reales — es solo para explorar la interfaz.

**Política ante meses faltantes**: si su serie tiene huecos (meses sin dato),
elija qué hacer:
- **Reportar**: no rellena nada, solo se lo avisa (recomendado si no está
  seguro de por qué falta el dato).
- **Interpolar**: rellena el hueco con una interpolación lineal en el tiempo.
- **Rellenar con cero**: asume que esos meses tuvieron demanda cero.

Después de cargar, la app le muestra:
- Un mensaje verde si todo salió bien (con el rango de fechas y el número de
  observaciones), o rojo con el detalle exacto de qué falló (columnas
  faltantes, meses no reconocidos, fechas duplicadas con valores distintos,
  etc.). La app **nunca adivina en silencio**: si detecta algo ambiguo
  (por ejemplo, si una coma en los números es separador decimal o de miles),
  se lo dice explícitamente en un cuadro azul de "Notas de carga".
- Una gráfica y una tabla con las primeras filas de su serie, para que
  confirme visualmente que se leyó bien.

Se requieren **al menos 18 observaciones** (meses) para continuar: con menos
no hay forma honesta de validar ningún método.

> ➡ Siguiente: [Módulo 2 — Clasificación y evaluación de métodos](#módulo-2--clasificación-y-evaluación-de-métodos)

### Módulo 2 — Clasificación y evaluación de métodos

Se ejecuta solo, apenas el Módulo 1 valida una serie. Muestra:

1. **Clasificación de la serie**: si tiene tendencia, si tiene
   estacionalidad, y si es estacionaria — con el resultado del test
   estadístico detrás de cada veredicto (no es una opinión, es un cálculo).
2. **Tabla de métodos evaluados**, ordenada por **MASE** (la métrica
   principal: valores menores a 1 significan que el método le gana al "naive
   estacional" — repetir el valor de hace 12 meses). La fila resaltada en
   verde es el método ganador. La tabla también muestra MAPE, MAD, ME
   (sesgo: positivo = el modelo subestima, negativo = sobreestima) y cuántos
   puntos de validación (`n`) sostienen cada número.
3. **Métodos excluidos**, con el motivo exacto (por ejemplo: "requiere 24
   observaciones y hay 20", o "no hay estacionalidad detectada"). Ningún
   método se descarta en silencio.
4. Un gráfico con el error de pronóstico (real − predicho) de los tres
   mejores métodos, fuera de muestra — es decir, sobre datos que el método NO
   vio al momento de ajustarse. Esa es la garantía central de esta
   herramienta: **ningún número que ve aquí se calculó sobre los mismos datos
   que se usaron para elegir el método o sus parámetros**.

No necesita hacer nada en este módulo salvo leer el resultado; el método
ganador pasa automáticamente al Módulo 3.

> ➡ Siguiente: [Módulo 3 — Pronóstico con intervalos](#módulo-3--pronóstico-con-intervalos)

### Módulo 3 — Pronóstico con intervalos

Elija:
- **Horizonte**: cuántos meses hacia adelante pronosticar (6, 12, 18 o 24).
- **Nivel del intervalo de predicción**: 80%, 90% o 95% — qué tan ancha
  quiere la banda de incertidumbre. Un intervalo del 95% es más ancho pero
  tiene más probabilidad de contener el valor real.

La gráfica muestra el histórico, el pronóstico (línea verde) y la banda
sombreada del intervalo elegido. La tabla debajo tiene los mismos números:
pronóstico, límites inferior/superior, sigma del error y sesgo estimado por
mes. El pronóstico nunca es negativo (se recorta en cero automáticamente).

> ➡ Siguiente: [Módulo 4 — Política de inventario](#módulo-4--política-de-inventario)

### Módulo 4 — Política de inventario

Indique:
- **Lead time (meses)**: cuánto tarda en llegar un pedido desde que se hace
  hasta que está disponible para vender.
- **Nivel de servicio**: qué tan seguro quiere estar de no quedarse sin
  stock durante el lead time (90%, 95%, 97.5% o 99% — más alto implica más
  stock de seguridad).

La app calcula automáticamente:
- **Demanda esperada durante el lead time.**
- **Stock de seguridad**: el colchón adicional para absorber la
  incertidumbre del pronóstico durante ese lead time.
- **Punto de reorden**: el nivel de inventario en el que debería lanzar un
  nuevo pedido.

Si el sesgo del pronóstico (ME) es grande frente a su variabilidad, la app
muestra un aviso explícito — es una señal de que el modelo está
sistemáticamente sobre o subestimando, y el punto de reorden debería
revisarse con más frecuencia.

> ➡ Siguiente (opcional): [Módulo 5 — Comparación externa](#módulo-5--comparación-externa-prophet--lightgbm)

### Módulo 5 — Comparación externa (Prophet / LightGBM)

Este módulo compara el método ganador de la Herramienta contra dos
pronosticadores externos muy usados en la industria: **Prophet** (Meta) y
**LightGBM** (gradient boosting, vía `mlforecast`). Es un módulo de
**referencia**, no de decisión: no cambia el método que usan los Módulos 3 y
4.

**Instalación (opcional, un paso adicional):**

```bash
pip install -r requirements-external.txt
```

Esto instala `prophet`, `cmdstanpy`, `mlforecast` y `lightgbm` — paquetes
más pesados que el resto de la herramienta, por eso no vienen instalados por
defecto. Si no los instala, el Módulo 5 aparece **deshabilitado** con un
aviso explicando cómo activarlo, y **el resto de la aplicación funciona
exactamente igual**.

**Cómo usarlo** (con `requirements-external.txt` instalado): cargue una
serie en el Módulo 1 (o use "Cargar datos de ejemplo") y presione
**"Ejecutar comparación externa"**. La app:

1. Evalúa la Herramienta, Prophet y LightGBM sobre el **mismo bloque de
   datos "fuera de muestra"** — ninguno de los tres vio esos meses al
   momento de ajustarse. Esto es intencional: mezclar el error interno de un
   método con el error externo de otro no es una comparación válida (así se
   documenta en `codigo/experimentos/decision_prophet.md`, sección de la
   Fase 11).
2. Muestra una tabla con MASE, MAPE, MAD, MSE y ME de cada método sobre ese
   bloque, y una gráfica con el histórico más el pronóstico de cada uno hacia
   adelante (mismo horizonte elegido en el Módulo 3).

Series muy cortas (menos de ~24-28 observaciones) pueden mostrar un aviso de
que no hay suficiente historia para el protocolo completo — es la misma
exigencia de honestidad estadística que rige el resto de la herramienta, no
un error.

## 4. Exportar resultados

En el Módulo 3, el botón **"Descargar forecast (Excel)"** genera un archivo
`.xlsx` con el pronóstico, los límites del intervalo, sigma del error y el
sesgo estimado por mes, para el horizonte y nivel que tenga seleccionados en
ese momento.

## 5. Procesar muchos productos a la vez (`batch_cli.py`)

Si tiene un catálogo de decenas o cientos de productos, no use la interfaz
web (piense en ella como la herramienta para explorar UN producto a la vez).
Use la línea de comandos:

```bash
python codigo/batch_cli.py productos.xlsx salida/ --horizon 12 --lead-time 3 --service-level 0.95
```

- `productos.xlsx`: mismo formato del Módulo 1, pero con la columna `sku`
  obligatoria (un producto por valor distinto de `sku`).
- `salida/`: carpeta donde se escriben `resumen_skus.csv` (una fila por
  producto: método ganador, MASE, MAPE, ME, etc.) y `pronosticos.csv` (el
  pronóstico completo de cada producto).
- La memoria usada **no crece** con el número de productos: los resultados se
  van guardando en disco a medida que se procesa cada uno, nunca se acumula
  el catálogo completo en memoria (importante si tiene un equipo con poca
  RAM).

Opciones útiles: `--gap-policy`, `--no-intervals`, `--no-inventory`,
`--flush-every N` (cada cuántos productos se vuelca a disco). Ejecute
`python codigo/batch_cli.py --help` para ver todas.

## 6. Preguntas frecuentes / problemas comunes

**"Faltan columnas obligatorias: sku"** — Está intentando filtrar por un
producto (`sku`) pero su archivo no tiene esa columna. Si es un solo
producto, no necesita la columna `sku`.

**"filas con mes no reconocido"** — Revise que la columna `month` tenga
números 1-12, o nombres de mes en español o inglés (completos o abreviados
a 3 letras: `ene`/`jan`, `dic`/`dec`, etc.).

**"Meses duplicados con valores DISTINTOS"** — Su archivo tiene el mismo mes
repetido con dos valores de demanda diferentes. La app no elige por usted;
corrija el archivo y vuelva a cargarlo.

**"Solo N observaciones útiles; se requieren al menos 18"** — La serie es
demasiado corta para una validación honesta. Consiga más historia o consulte
con quien mantiene el proyecto si su caso justifica una excepción.

**Un método aparece en "Métodos excluidos"** — No es un error: la app
excluye explícitamente los métodos que no pueden evaluarse con honestidad
sobre su serie (por ejemplo, un modelo estacional sin suficientes ciclos de
historia). El motivo exacto siempre se muestra junto al método.

**El Módulo 5 dice "deshabilitado"** — No tiene instalado
`requirements-external.txt`. Es opcional; instálelo solo si necesita
comparar contra Prophet/LightGBM (ver la sección del Módulo 5 arriba).

**Quiero entender qué corrige esta versión de la herramienta frente a la
original** — Ese es el contenido de `CHANGELOG.md` y `RESUMEN_EJECUCION.md`
en la raíz del repositorio; este manual solo cubre el uso de la aplicación.

**Quiero reproducir las cifras del manuscrito académico** — Vea la sección
["Reproducir los resultados del manuscrito"](../README.md#reproducir-los-resultados-del-manuscrito)
del `README.md`: cada cifra sale de un script en `codigo/experimentos/`,
ejecutable con un solo comando.

---

[↑ Volver al índice del manual](#índice) · [⬅ Volver al README](../README.md)
