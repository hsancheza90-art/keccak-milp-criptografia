# Estado del avance

## Etapa 1. Preparación del entorno

Completado:

- creación del entorno virtual;
- instalación de dependencias;
- configuración de CBC;
- instalación editable del paquete;
- configuración de `pytest`.

## Etapa 2. Estado y permutaciones

Completado:

- representación del estado como $5\times5\times z$;
- implementación y validación de $\rho$;
- implementación y validación de $\pi$;
- implementación de la composición $\rho+\pi$;
- validaciones para $z=4$ y $z=8$.

## Etapa 3. Esqueleto MILP

Completado:

- creación de estados de frontera;
- variables binarias;
- entrada diferencial no nula;
- objetivo provisional;
- diferenciación entre variables declaradas y conectadas;
- validación de seis configuraciones.

## Etapa 4. Capa $\theta$

Completado:

- implementación funcional con NumPy;
- cálculo de paridades $C$;
- cálculo del efecto $D$;
- validación de linealidad;
- formulación MILP exacta;
- variables auxiliares de paridad;
- comparación NumPy versus MILP;
- pruebas para $z=4$ y $z=8$.

## Etapa 5. Capas $\rho$ y $\pi$ en MILP

Completado:

- variables de salida;
- restricciones de permutación;
- integración posterior a $\theta$;
- validación estructural;
- pruebas automáticas.

## Estado de pruebas

Todas las pruebas implementadas hasta el momento terminan correctamente.

## Próxima etapa

Implementar la capa no lineal $\chi$:

- versión funcional;
- representación diferencial;
- variables AND auxiliares;
- variables de actividad de S-box;
- función objetivo para minimizar S-boxes activas.