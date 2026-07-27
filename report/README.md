# Plantilla de informe académico

Esta carpeta contiene una plantilla minimalista en LaTeX para el informe:

**Modelado MILP de Keccak reducido para la minimización de S-boxes activas**

## Estructura

```text
report/
├── main.tex
├── references.bib
├── build.ps1
├── README.md
├── figures/
└── sections/
    ├── 01_introduccion.tex
    ├── 02_fundamentos.tex
    ├── 03_metodologia.tex
    ├── 04_resultados.tex
    ├── 05_reproducibilidad.tex
    └── 06_conclusiones.tex
```

## Ubicación recomendada

Copie la carpeta `report/` en la raíz del repositorio:

```text
PracticaCalificada/
├── report/
├── scripts/
├── src/
├── tests/
└── outputs/
```

La figura se carga automáticamente desde:

```text
../outputs/figures/active_sbox_bounds.png
```

## Compilación

Desde la carpeta `report/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

O ejecute:

```powershell
.\build.ps1
```

Para limpiar archivos auxiliares:

```powershell
latexmk -C
```

## Actualizar resultados antes de compilar

Desde la raíz del repositorio:

```powershell
python scripts/run_active_sbox_experiments.py --mode quick
```

## Criterio de redacción

La plantilla distingue deliberadamente:

- resultados exactos para una y dos rondas;
- cotas y testigos para tres rondas;
- resultados globales y resultados restringidos a la familia `2+2+c`.
