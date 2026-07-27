# Modelado MILP de Keccak reducido

Implementación reproducible de un modelo de Programación Lineal Entera Mixta
(MILP) para estudiar el número mínimo de aplicaciones activas de la capa no
lineal $\chi$ en versiones reducidas de Keccak.

El proyecto fue desarrollado como parte del curso:

> **MCC640B — Tópicos Especiales IV: Criptoanálisis, Codificación y Seguridad**  
> Maestría en Ciencias de la Computación  
> Universidad Nacional de Ingeniería

---

## Objetivo

Modelar bit a bit una versión reducida de Keccak mediante MILP y determinar,
o cuando no sea posible certificarlo exactamente, acotar el número mínimo de
S-boxes activas durante una, dos y tres rondas.

El modelo representa las transformaciones:

- $\theta$: difusión entre columnas.
- $\rho$: rotación de las palabras.
- $\pi$: permutación de posiciones.
- $\chi$: capa no lineal.
- $\iota$: incorporación de la constante de ronda.

La formulación utiliza dos ejecuciones simultáneas de Keccak y calcula sus
diferencias mediante restricciones XOR exactas.

---

## Configuraciones estudiadas

El estado de Keccak se representa como:

$$
A[x,y,k] \in \{0,1\},
$$

con:

$$
x,y \in \{0,\ldots,4\},
\qquad
k \in \{0,\ldots,z-1\}.
$$

El tamaño total del estado es:

$$
25z \text{ bits}.
$$

Las configuraciones evaluadas son:

| $z$ | Tamaño del estado | Palabras | Aplicaciones de $\chi$ por ronda |
|---:|---:|---:|---:|
| 4 | 100 bits | 25 | 20 |
| 8 | 200 bits | 25 | 40 |

Cada aplicación local de $\chi$ procesa cinco bits que comparten las
coordenadas $(y,k)$. Una S-box se considera activa cuando recibe una
diferencia de entrada no nula.

---

## Correspondencia experimental entre intentos y rondas

La actividad académica relaciona el número de intentos con tres escenarios:

| Intentos | Rondas analizadas |
|---|---:|
| Menos de 10 | 1 |
| Entre 10 y 19 | 2 |
| Entre 20 y 29 | 3 |

Esta correspondencia se utiliza únicamente como criterio experimental para
seleccionar el número de rondas. El número de intentos no modifica las
transformaciones internas de Keccak ni constituye una relación criptográfica
propia del estándar.

---

## Modelo MILP

El modelo contiene dos ejecuciones del estado:

$$
A_r^L
\qquad \text{y} \qquad
A_r^R.
$$

La diferencia se calcula mediante:

$$
\Delta A_r = A_r^L \oplus A_r^R.
$$

Para una XOR binaria:

$$
d = u \oplus v,
$$

se introduce una variable auxiliar binaria $q$ y se impone:

$$
u+v=d+2q.
$$

La capa $\chi$ se representa exactamente mediante la linealización del
producto binario presente en su función booleana.

Para cada ronda $r$ y posición local $(y,k)$ se define:

$$
a_{r,y,k} \in \{0,1\},
$$

donde:

$$
a_{r,y,k}
=
\bigvee_{x=0}^{4}
\Delta B_r[x,y,k].
$$

La función objetivo minimiza la actividad total:

$$
\min
\sum_{r=0}^{R-1}
\sum_{y=0}^{4}
\sum_{k=0}^{z-1}
a_{r,y,k}.
$$

También se impone una diferencia inicial no nula para evitar la solución
trivial:

$$
\sum_{x=0}^{4}
\sum_{y=0}^{4}
\sum_{k=0}^{z-1}
\Delta A_{0,x,y,k}
\geq 1.
$$

---

## Resultados principales

| $z$ | Rondas | Cota inferior | Cota superior | Testigo | Interpretación |
|---:|---:|---:|---:|---:|---|
| 4 | 1 | 1 | 1 | $1$ | Óptimo global |
| 4 | 2 | 4 | 4 | $2+2$ | Óptimo global |
| 4 | 3 | 5 | 13 | $2+2+9$ | Óptimo restringido |
| 8 | 1 | 1 | 1 | $1$ | Óptimo global |
| 8 | 2 | 4 | 4 | $2+2$ | Óptimo global |
| 8 | 3 | 5 | 14 | $2+2+10$ | Óptimo restringido |

### Una ronda

$$
N_{\mathrm{act}}^{\min}(4,1)
=
N_{\mathrm{act}}^{\min}(8,1)
=
1.
$$

### Dos rondas

$$
N_{\mathrm{act}}^{\min}(4,2)
=
N_{\mathrm{act}}^{\min}(8,2)
=
4.
$$

Los testigos óptimos presentan la distribución:

$$
2+2.
$$

### Tres rondas

$$
5
\leq
N_{\mathrm{act}}^{\min}(4,3)
\leq
13,
$$

y:

$$
5
\leq
N_{\mathrm{act}}^{\min}(8,3)
\leq
14.
$$

Los mejores testigos tienen distribuciones:

$$
2+2+9
$$

para $z=4$, y:

$$
2+2+10
$$

para $z=8$.

Los valores 13 y 14 son óptimos dentro de la familia restringida:

$$
2+2+c,
$$

pero no constituyen mínimos globales certificados para el espacio completo
de tres rondas.

---

## Búsqueda restringida de tres rondas

| $z$ | Trayectorias | Realizaciones por trayectoria | Total evaluado |
|---:|---:|---:|---:|
| 4 | 200 | 1024 | 204 800 |
| 8 | 400 | 1024 | 409 600 |

Cada S-box de cinco bits admite:

$$
2^5=32
$$

valores absolutos posibles. Al existir dos S-boxes activas en la segunda
ronda, se evalúan:

$$
32^2=1024
$$

realizaciones por trayectoria.

La búsqueda es exhaustiva dentro de la familia $2+2+c$, pero no dentro del
espacio completo de trayectorias de tres rondas.

---

## Estado del proyecto

El proyecto cuenta actualmente con:

- Implementación funcional de $\theta$, $\rho$, $\pi$, $\chi$ e $\iota$.
- Composición de una o varias rondas de Keccak reducido.
- Formulación MILP exacta de las cinco transformaciones.
- Dos ejecuciones simultáneas y diferencias XOR exactas.
- Variables para el conteo de S-boxes activas.
- Función objetivo global.
- Restricciones de actividad total y actividad por ronda.
- Fijación de soportes y estados concretos.
- Testigos reconstruibles.
- Certificación de mínimos globales para una y dos rondas.
- Búsqueda exhaustiva restringida para tres rondas.
- Generación automática de resultados en CSV y JSON.
- Figura comparativa de cotas.
- Notebook final ejecutable.
- Informe académico en LaTeX.
- Control de versiones mediante Git.
- 245 pruebas automatizadas aprobadas.

---

## Estructura del repositorio

~~~text
PracticaCalificada/
│
├── src/
│   └── keccak_milp/
│
├── scripts/
│   └── run_active_sbox_experiments.py
│
├── tests/
│
├── notebooks/
│   └── keccak_milp_active_sboxes.ipynb
│
├── outputs/
│   ├── results/
│   │   ├── active_sbox_results.csv
│   │   ├── active_sbox_witnesses.json
│   │   ├── active_sbox_summary.md
│   │   └── experiment_environment.json
│   └── figures/
│       └── active_sbox_bounds.png
│
├── report/
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   ├── build.ps1
│   ├── figures/
│   └── sections/
│       ├── 01_introduccion.tex
│       ├── 02_keccak_reducido.tex
│       ├── 03_formulacion_milp.tex
│       ├── 04_implementacion.tex
│       ├── 05_metodologia_experimental.tex
│       ├── 06_resultados.tex
│       ├── 07_reproducibilidad.tex
│       ├── 08_limitaciones.tex
│       ├── 09_conclusiones.tex
│       └── 10_disponibilidad_codigo.tex
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
~~~

---

## Instalación

Se recomienda utilizar Python 3.12 y un entorno virtual.

En PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

La implementación utiliza Python, NumPy, PuLP, CBC y pytest.

---

## Ejecución de pruebas

Desde la raíz del repositorio:

~~~powershell
python -m pytest -q --disable-warnings
~~~

Resultado esperado:

~~~text
245 passed
~~~

---

## Ejecución de experimentos

### Modo rápido

~~~powershell
python scripts/run_active_sbox_experiments.py --mode quick
~~~

### Modo completo

~~~powershell
python scripts/run_active_sbox_experiments.py --mode full
~~~

---

## Compilación del informe

Desde la carpeta `report/`:

~~~powershell
latexmk `
    -pdf `
    -interaction=nonstopmode `
    -halt-on-error `
    -file-line-error `
    main.tex
~~~

También puede utilizarse:

~~~powershell
.\build.ps1
~~~

La bibliografía se gestiona mediante `biblatex` y Biber. El documento
compilado se encuentra en `report/main.pdf`.

---

## Reproducibilidad

Los resultados se validan mediante:

1. Implementación funcional independiente.
2. Comparación bit a bit con el modelo MILP.
3. Reconstrucción de los estados iniciales.
4. Recálculo de las diferencias antes de $\chi$.
5. Reconstrucción de los soportes activos.
6. Comparación con la actividad reportada.
7. Pruebas automatizadas.
8. Artefactos estructurados en CSV y JSON.

La versión experimental de referencia está identificada mediante la etiqueta:

~~~text
keccak-milp-experiments-v1
~~~

---

## Alcance y limitaciones

Los resultados corresponden a Keccak reducido con:

$$
z\in\{4,8\}
$$

y un máximo de tres rondas.

No deben interpretarse como una evaluación completa de la seguridad de
SHA-3 ni como resultados aplicables directamente a Keccak-$f[1600]$.

El modelo actual no calcula directamente:

- Probabilidades diferenciales.
- Pesos diferenciales de las transiciones de $\chi$.
- Correlaciones lineales.
- Complejidad de ataques contra SHA-3.
- Mínimos globales exactos para tres rondas.

---

## Trabajo futuro

- Explorar familias de tres rondas distintas de $2+2+c$.
- Incorporar ruptura de simetrías.
- Fortalecer la formulación mediante desigualdades válidas.
- Comparar CBC con Gurobi.
- Incorporar pesos diferenciales de $\chi$.
- Analizar tamaños de palabra y números de rondas mayores.
- Cerrar las cotas globales de tres rondas.

---

## Informe

El informe académico completo se encuentra en:

~~~text
report/main.pdf
~~~
