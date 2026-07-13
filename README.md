# Modelo MILP de Keccak modificado

Implementación de un modelo de Programación Lineal Entera Mixta para estudiar el número mínimo de S-boxes activas en versiones reducidas de Keccak.

El proyecto corresponde a una práctica calificada del curso de Criptografía y Criptoanálisis.

## Objetivo

Modelar mediante MILP las transformaciones de Keccak:

- $\theta$
- $\rho$
- $\pi$
- $\chi$

y encontrar el número mínimo de S-boxes activas para:

- $z=4$
- $z=8$
- 1, 2 y 3 rondas

La variable dinámica `intentos` se relaciona experimentalmente con el número de rondas:

| Intentos | Rondas |
|---|---:|
| Menos de 10 | 1 |
| Menos de 20 | 2 |
| Menos de 30 | 3 |

## Dimensiones del estado

Keccak utiliza un estado de:

$$
5 \times 5 \times z
$$

Para los escenarios evaluados:

| z | Tamaño del estado | S-boxes $\chi$ por ronda |
|---:|---:|---:|
| 4 | 100 bits | 20 |
| 8 | 200 bits | 40 |

## Estado actual

Se han implementado y validado:

- configuración del entorno;
- integración con CBC;
- estructura modular del proyecto;
- representación del estado $5\times5\times z$;
- implementación funcional de $\rho$;
- implementación funcional de $\pi$;
- implementación funcional de $\theta$;
- formulación MILP exacta de $\theta$;
- formulación MILP de $\rho$ y $\pi$;
- pruebas automáticas con `pytest`;
- notebooks reproducibles;
- comparación entre NumPy y MILP.

Pendiente:

- implementación funcional y MILP de $\chi$;
- conteo de S-boxes activas;
- conexión entre rondas;
- experimentos para $z=4,8$ y 1, 2 y 3 rondas;
- comparación CBC/Gurobi;
- cálculo de probabilidades diferenciales;
- elaboración del informe final.

## Estructura

```text
PracticaCalificada/
│
├── docs/
├── notebooks/
│   ├── 00_validacion_entorno.ipynb
│   ├── 01_estado_rho_pi.ipynb
│   ├── 02_esqueleto_modelo_milp.ipynb
│   ├── 03_capa_theta.ipynb
│   └── 04_theta_milp.ipynb
├── results/
│   ├── figures/
│   ├── logs/
│   ├── raw/
│   └── tables/
├── scripts/
│   ├── run_experiments.py
│   └── validate_environment.py
├── src/
│   └── keccak_milp/
│       ├── __init__.py
│       ├── config.py
│       ├── layers.py
│       ├── model.py
│       ├── results.py
│       └── solver.py
├── tests/
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-gurobi.txt