# Especificación compacta del caso final de Keccak

## 1. Identidad del trabajo

- Curso: MCC640B - Tópicos Especiales IV: Criptoanálisis, Codificación y Seguridad.
- Caso: rediseño experimental de Keccak reducido para hardware limitado.
- Línea base: commit `4d442d67eda84cd14533ca3c7f92e82d04b02730`.
- Rama de desarrollo: `feature/examen-final-keccak`.
- Tamaños de palabra: `z = 4` y `z = 8`.
- Rondas experimentales: 1, 2 y 3.

## 2. Objetivo

Diseñar y evaluar una variante compacta de Keccak reducido que incorpore dos parámetros públicos dinámicos, mejore la cobertura de difusión y reduzca o mantenga el costo de la capa no lineal, comparándola con la V1 mediante análisis estructural y MILP.

La propuesta es exclusivamente académica y no se presenta como sustituto de SHA-3 ni como construcción segura para producción.

## 3. Línea base V1

La línea base conserva la ronda:

`theta -> rho -> pi -> chi -> iota`

Resultados auditados:

| z | Rondas | Resultado |
|---:|---:|:---|
| 4 | 1 | Mínimo global exacto: 1 |
| 4 | 2 | Mínimo global exacto: 4 |
| 4 | 3 | Cota global: 5 a 13 |
| 8 | 1 | Mínimo global exacto: 1 |
| 8 | 2 | Mínimo global exacto: 4 |
| 8 | 3 | Cota global: 5 a 14 |

Los valores 13 y 14 son testigos de una búsqueda restringida y no mínimos globales demostrados.

La V1 dispone de 245 pruebas automatizadas superadas.

## 4. Parámetros dinámicos

La variante utilizará dos parámetros públicos:

### 4.1 Nivel de seguridad

`security_level` seleccionará un perfil previamente definido:

| Valor | Perfil | Rondas |
|---:|:---|---:|
| 0 | Ligero | 1 |
| 1 | Equilibrado | 2 |
| 2 | Reforzado | 3 |

El parámetro no dependerá de la clave ni de información secreta.

### 4.2 Dominio de aplicación

`domain_id` identificará el contexto de uso:

| Valor | Dominio experimental |
|---:|:---|
| 0 | Hash general |
| 1 | Autenticación |
| 2 | Derivación de claves |
| 3 | Integridad de firmware |

El dominio se incorporará mediante una constante pública de separación y no mediante ramas secretas ni rotaciones arbitrarias.

## 5. Variante propuesta

La ronda modificada tendrá la forma general:

`L* -> chi* -> iota*`

donde:

- `L*` será una capa lineal basada en XOR, rotaciones y permutaciones fijas.
- `chi*` será una modificación controlada de la capa no lineal.
- `iota*` incorporará la constante de ronda y la separación de dominio.

La V1 no será alterada. La variante se implementará en módulos separados.

## 6. Evaluación de la difusión

### 6.1 Cobertura estructural

Para cada bit de entrada se calculará qué posiciones de salida dependen de él.

La cobertura después de `r` aplicaciones será:

`Cov_r(i) = |Reach_r(i)| / (25z)`.

El criterio principal será:

`min_i Cov_r(i) >= 0.80`.

El objetivo es alcanzar este umbral en menos aplicaciones o rondas que la línea base.

### 6.2 Efecto avalancha

También se medirán diferencias reales entre estados que difieren en un bit.

La proporción esperada para un comportamiento equilibrado será cercana a 0.50, no a 0.80.

Se reportarán mínimo, promedio y máximo sobre un conjunto reproducible de estados.

## 7. Evaluación de la capa no lineal

Se compararán como mínimo:

- Operaciones AND.
- Operaciones XOR.
- Operaciones NOT.
- Profundidad lógica estimada.
- Biyectividad.
- Uniformidad diferencial.
- Correlación lineal máxima.

La variante solo será aceptada si la reducción de costo no introduce transiciones diferenciales deterministas ni aproximaciones lineales perfectas no triviales.

## 8. Evaluación MILP

El modelo minimizará el número total de componentes no lineales activos:

`min sum(a[r,j])`.

Se conservarán:

- Restricción de diferencia inicial no nula.
- Variables binarias de actividad.
- Experimentos para `z = 4, 8`.
- Evaluación de 1, 2 y 3 rondas.

Se compararán la V1 y la variante usando el mismo criterio de actividad.

## 9. Vulnerabilidades que deben analizarse

La solución documentará al menos dos riesgos:

1. Cancelación o pérdida de difusión causada por parámetros dinámicos mal elegidos.
2. Filtración temporal si el número de rondas o el flujo depende de información secreta.

Cada vulnerabilidad deberá acompañarse de un contraejemplo y una corrección.

## 10. Entregables mínimos

- Implementación separada de la variante.
- Pruebas automatizadas del caso final.
- Resultados de difusión.
- Resultados MILP.
- Tabla comparativa V1-variante.
- Dos vulnerabilidades y sus correcciones.
- Informe final compacto en LaTeX.
- PDF compilado.

## 11. Elementos fuera de alcance

- Rediseñar Keccak-f[1600].
- Proponer una alternativa de producción a SHA-3.
- Cerrar los mínimos globales pendientes de tres rondas de la V1.
- Migrar toda la base histórica a PuLP 4.0.
- Repetir los doce notebooks progresivos de la V1.
- Elaborar nuevamente un informe de cincuenta páginas.

## 12. Criterios de aceptación

El caso final se considerará completo cuando:

1. Las 245 pruebas históricas continúen superándose.
2. Los parámetros dinámicos estén validados y sean públicos.
3. La variante conserve la forma y binariedad del estado.
4. La nueva capa lineal sea lineal e invertible.
5. La cobertura estructural mínima alcance al menos 80 %.
6. La capa no lineal tenga análisis DDT, LAT y costo lógico.
7. El MILP produzca resultados reproducibles.
8. La comparación no presente cotas como mínimos exactos.
9. Git permanezca limpio después de generar los artefactos finales.
