# Resumen de ejecución — Fase 13 (anonimización repo-wide, 2026-09-01)

## Contexto

Una auditoría independiente posterior a la Fase 12 confirmó que el código
funcional está correcto y que `manuscritos/articulo_mdpi/template.tex` ya
está anonimizado (`grep -ic <nombre-real-de-la-empresa> template.tex` = 0).
Sin embargo, esa auditoría encontró que el **repositorio público** seguía
conteniendo el nombre real de la empresa en 13 archivos. Esto es grave porque
el Data Availability Statement del manuscrito remite directamente a la URL
pública de este repositorio, así que cualquier revisor que abra ese enlace
ve el nombre real pese a que el `.tex` está anonimizado. Esta fase cierra
esa fuga.

## Paso 0 — Diagnóstico

Comando ejecutado (`<nombre-real-de-la-empresa>` es un marcador de posición
por el nombre real de la empresa, para no reintroducir la fuga dentro de
este mismo documento):

```
grep -rli <nombre-real-de-la-empresa> --include="*" . | grep -v "^./.git/" | grep -v "__pycache__"
```

Resultado (13 archivos, coincide exactamente con la lista del prompt de
esta fase):

```
./CHANGELOG.md
./codigo/app.py
./codigo/experimentos/caso_ilustrativo.py
./codigo/experimentos/comparativa_externa.py
./codigo/experimentos/decision_prophet.md
./codigo/experimentos/vs_incumbente.py
./docs/MANUAL_USUARIO.md
./docs/prompt_maestro.md
./docs/prompt_maestro_fase11.md
./README.md
./resultados/logs/vs_incumbente.log
./RESUMEN_EJECUCION.md
./RESUMEN_EJECUCION_FASE12.md
```

No se encontraron ocurrencias adicionales fuera de esta lista (ni en datos,
notebooks, ni configuración).

## Paso 1 — Archivos modificados

| Archivo | Líneas cambiadas | Tipo |
|---|---|---|
| `README.md` | 4 (título, subtítulo de sección, limitación, ancla de enlace) | b (título) + a (narrativo) |
| `codigo/app.py` | 2 (comentario de cabecera, `app.title`) | b |
| `docs/MANUAL_USUARIO.md` | 1 (título) | b |
| `codigo/experimentos/caso_ilustrativo.py` | 2 (docstring) | a |
| `codigo/experimentos/comparativa_externa.py` | 1 (comentario) | a |
| `codigo/experimentos/decision_prophet.md` | 1 (prosa) | a |
| `codigo/experimentos/vs_incumbente.py` | 5 (docstring, help de `--input`, mensajes impresos) | a |
| `resultados/logs/vs_incumbente.log` | regenerado (no editado a mano) | c |
| `docs/prompt_maestro.md` | 6 (título del documento + 5 en prosa) | a |
| `docs/prompt_maestro_fase11.md` | 1 (prosa) | a |
| `RESUMEN_EJECUCION.md` | 2 (prosa) | a |
| `CHANGELOG.md` | 2 (1 narrativa F09 + 1 referencia dentro de comando `grep` citado en F41) | a |
| `RESUMEN_EJECUCION_FASE12.md` | 2 (referencias dentro de comandos `grep` citados en F40/F41) | a |

**Notas de criterio aplicado:**

- **Tipo b** (`README.md` título, `docs/MANUAL_USUARIO.md` título,
  `codigo/app.py`): se trata como "nombre del proyecto" — la referencia a
  la empresa se **eliminó**, no se reemplazó por "la empresa de
  referencia" (sería extraño como título de documento o de pestaña del
  navegador). Se verificó que no había anclas de enlace internas (`#...`)
  apuntando a los títulos modificados, salvo en `README.md` donde sí había
  una (`↑ Volver al inicio`), que se actualizó junto con el título.
- **Tipo a**: reemplazo narrativo estándar del nombre real de la empresa
  por "la empresa de referencia" (todo el contexto es español en estos 13
  archivos; no hubo contexto en inglés fuera de `template.tex`, que no se
  tocó).
- **CHANGELOG.md / RESUMEN_EJECUCION_FASE12.md — caso especial**: tres
  líneas citan literalmente el comando `grep -ic` con el nombre real de la
  empresa que se corrió en la Fase 12 para verificar la anonimización del
  `.tex`. Reemplazar el nombre real por "la empresa de referencia" ahí
  habría sido engañoso (el
  comando real que se corrió en su momento buscaba el nombre real, no la
  frase de reemplazo). Se reformularon como `` `grep -ic` del nombre real
  de la empresa `` (sin nombrarla), preservando el significado histórico
  del check (0 ocurrencias) sin reintroducir la fuga.

## Paso 1(c) — Regeneración del log

`resultados/logs/vs_incumbente.log` es la salida de una corrida anterior de
`codigo/experimentos/vs_incumbente.py`. Se corrigió primero el `.py`
(commit `6f27511`), y luego se regeneró el log corriendo:

```
python codigo/experimentos/vs_incumbente.py --synthetic --n-series 40 --seed 20260824
```

(mismos parámetros citados en `README.md` y en el manuscrito, Tabla
`tab:incumbente`). Antes de regenerar se verificaron las cifras que el
manuscrito cita textualmente de esta corrida
(`manuscritos/articulo_mdpi/template.tex`, Sección 3.3):

| Cifra citada en el manuscrito | Valor pre-regeneración (log viejo) | Valor post-regeneración | ¿Coincide? |
|---|---|---|---|
| MASE mediano (incumbente/naive/herramienta) | 0.922 / 0.949 / 0.807 | 0.922 / 0.949 / 0.807 | Sí |
| MAPE mediano (%) | 14.4 / 11.4 / 10.1 | 14.4 / 11.4 / 10.1 | Sí |
| \|ME\| mediano | 217.1 / 55.4 / 111.9 | 217.1 / 55.4 / 111.9 | Sí |
| Supera al incumbente | 32/40 (80%) | 32/40 (80%) | Sí |
| Mejora mediana por serie | +19.9% | +19.9% | Sí |
| Mejora de las medianas | +12.5% | +12.5% | Sí |
| Wilcoxon (herramienta vs. incumbente) | — (no estaba en el log viejo) | W=94.0, p=4.93e-06, n=40 | Coincide con el manuscrito (el log viejo era de una versión del script anterior a F34, que no imprimía este bloque) |
| Desglose vs. naive (victorias/empates/derrotas) | — (no estaba en el log viejo) | 24/10/6 | Coincide con el manuscrito |

Ningún número cambió; la regeneración solo añadió las secciones de Wilcoxon
y desglose W/T/L que el script ya calculaba (desde F34, Fase 12) pero que no
estaban en esa corrida archivada del log. `resultados/vs_incumbente.csv` y
`resultados/vs_incumbente_resumen.csv` (los CSV que sí están fuera de
alcance de esta fase) se regeneraron como efecto colateral de correr el
script, pero `git status`/`git diff` confirmó **cero diferencias** — no fue
necesario descartar ni restaurar nada.

## Paso 2 — Verificación

**Check 1** — `grep -rli <nombre-real-de-la-empresa> --include="*" . | grep -v "^./.git/" | grep -v "__pycache__"`

```
(sin salida — vacío)
```

**Check 2** — `grep -ic <nombre-real-de-la-empresa> manuscritos/articulo_mdpi/template.tex`

```
0
```

(sin cambios respecto al estado previo a esta fase, como se esperaba —
`template.tex` no fue tocado)

**Check 3** — `pytest codigo/tests -q` (vía `--junitxml` para un conteo
exacto, dado que la captura de consola en este entorno Windows no imprimía
la línea de resumen final)

```
tests=276  failures=0  errors=0  skipped=1
```

276 tests totales, 0 fallos, 0 errores — coincide con el criterio de
aceptación. El conteo de *skipped* difiere del "12 skipped" citado en el
contexto de esta tarea: en este entorno `prophet`, `mlforecast` y
`lightgbm` están instalados, por lo que los `pytest.importorskip(...)` de
`codigo/tests/test_external_baselines.py` no saltan ninguna prueba aquí (a
diferencia de un entorno sin esos paquetes opcionales). El único skip
real (`test_no_leakage.py::test_historia_insuficiente_levanta_excepcion_no_devuelve_otro_modelo[naive]`,
"naive si puede operar con 2 observaciones") es un skip condicional
esperado del propio test, no una prueba nueva ni una regresión. Esto
además coincide exactamente con el propio registro de la Fase 12
(`RESUMEN_EJECUCION_FASE12.md`, línea 30-33: "276 después (275 passed, 1
skipped, exit 0)"), confirmando que 275 passed / 1 skipped / 0 failed es el
baseline correcto de este repositorio, y no hubo ninguna regresión
introducida por los cambios de esta fase. Ningún test necesitó
modificarse: ningún test compara literalmente un docstring o help text que
haya cambiado.

**Check 4** — título de la interfaz (`codigo/app.py`)

```
$ grep -n "app.title\|# app.py" codigo/app.py
2:# app.py - Motor de Pronosticos (interfaz Dash)
42:app.title = "Motor de Pronosticos"
```

Se verificó además que el archivo sigue siendo sintácticamente válido
(`ast.parse` sobre `codigo/app.py`, OK) tras los cambios. No se lanzó el
servidor Dash interactivo (quedaría escuchando indefinidamente en
`127.0.0.1:8050`); la confirmación de que el título ya no contiene el
nombre real de la empresa se hizo por grep directo sobre la línea de
`app.title`, que es donde Dash toma el texto de la pestaña del navegador.

## Confirmación: ningún cambio tocó lógica de negocio

Los 13 archivos modificados en esta fase contienen exclusivamente cambios
de **strings de texto**: títulos de documento, docstrings, comentarios,
mensajes impresos por `vs_incumbente.py`, y prosa en archivos `.md`. No se
modificó ninguna firma de función, ningún valor numérico, ningún parámetro
por defecto, ninguna fórmula ni ningún flujo de control. La única
excepción aparente — la regeneración de `resultados/logs/vs_incumbente.log`
— es también un cambio puramente de reproducción (mismo script, mismos
parámetros, mismas cifras), no de lógica.

## Nota sobre el historial de git (fuera de alcance)

El nombre real de la empresa sigue presente en commits anteriores a esta
fase (por ejemplo, en los mensajes de commit o en versiones previas de los
13 archivos listados arriba). Reescribir el historial de git (`git
filter-repo`, `git filter-branch`, o recrear el repositorio desde un
squash) eliminaría esa exposición, pero es una operación destructiva e
irreversible que reescribe SHAs, invalida cualquier fork/clon existente, y
requeriría coordinación fuera de esta fase (por ejemplo, si el repositorio
ya fue clonado por revisores). Se deja fuera de alcance de la Fase 13 tal
como indica el prompt; si se decide limpiar el historial, debe confirmarse
explícitamente antes de ejecutarlo.
