# Resultados de S-boxes activas

Modo de ejecución: `quick`.

| z | Rondas | Intentos | Estado | Mínimo | Cota inferior | Cota superior | Testigo |
|---:|---:|:---:|:---|---:|---:|---:|:---|
| 4 | 1 | <10 | exact | 1 | 1 | 1 | `1` |
| 4 | 2 | <20 | exact | 4 | 4 | 4 | `2+2` |
| 4 | 3 | <30 | bounded | — | 5 | 13 | `2+2+9` |
| 8 | 1 | <10 | exact | 1 | 1 | 1 | `1` |
| 8 | 2 | <20 | exact | 4 | 4 | 4 | `2+2` |
| 8 | 3 | <30 | bounded | — | 5 | 14 | `2+2+10` |

## Interpretación

- Una y dos rondas tienen mínimos globales exactos.
- Para tres rondas se informa una cota inferior global y el mejor testigo validado.
- Los resultados de tres rondas están restringidos a la familia `2+2+c`.
- Los testigos fueron contrastados mediante el modelo MILP exacto y la implementación de referencia.
