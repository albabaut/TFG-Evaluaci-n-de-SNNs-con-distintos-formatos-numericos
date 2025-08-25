# Análisis de RMSE

Este directorio contiene figuras relacionadas con el análisis de Error Cuadrático Medio (RMSE) entre diferentes formatos de precisión numérica.

## Contenido

### RMSE vs Corriente (I)
- **rmse_vs_I.png** - RMSE vs corriente de entrada para diferentes formatos
- **rmse_vs_I_extended.png** - Análisis extendido de RMSE vs corriente
- **rmse_vs_I_p24.png** - RMSE vs corriente para precisión de 24 bits
- **rmse_vs_I_p32.png** - RMSE vs corriente para precisión de 32 bits
- **rmse_vs_I_p24_p20.png** - RMSE vs corriente comparando 24 bits y 20 bits
- **rmse_vs_I_bfloat.png** - RMSE vs corriente para formato bfloat
- **rmse_vs_I_golden.png** - RMSE vs corriente para referencia dorada
- **rmse_vs_I_cercacero.png** - RMSE vs corriente cerca de valores cero

### Mapas de Calor de RMSE
- **heatmap_rmse_posit16.png** - Mapa de calor de RMSE para formato posit16
- **heatmap_rmse_float32.png** - Mapa de calor de RMSE para formato float32
- **heatmap_rmse_float16.png** - Mapa de calor de RMSE para formato float16

### Otras Métricas de RMSE
- **rmse_spike_vs_current.png** - RMSE del timing de spikes vs corriente
- **rmse_barplot_red.png** - Gráfico de barras de valores de RMSE
- **rmse_vs_corriente.png** - RMSE vs corriente (etiquetado en español)

## Descripción

Estas figuras analizan la precisión numérica de diferentes formatos de precisión midiendo el Error Cuadrático Medio entre los resultados de simulación y los valores de referencia. Un RMSE más bajo indica mejor precisión numérica. 