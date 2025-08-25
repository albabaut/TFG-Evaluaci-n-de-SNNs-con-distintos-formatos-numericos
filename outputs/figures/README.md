# Directorio de Figuras

Este directorio contiene todas las figuras y visualizaciones generadas del proyecto de simulación de neurociencia.

## Estructura del Directorio

### 📊 **Categorías Principales de Análisis**

- **`precision_comparisons/`** - Figuras comparando diferentes formatos de precisión numérica
- **`rmse_analysis/`** - Análisis de Error Cuadrático Medio (RMSE) entre formatos de precisión
- **`spike_analysis/`** - Análisis de timing de spikes, tasas de disparo y comportamiento de spiking
- **`error_analysis/`** - Análisis de acumulación y distribución de errores
- **`simulation_results/`** - Resultados directos de simulaciones
- **`metrics_analysis/`** - Métricas de calidad y medidas de comparación

### 📁 **Directorios Especializados**

- **`figuras/`** - Figuras principales de análisis (organizadas por tipo de análisis)
- **`images/`** - Imágenes generales de simulación
- **`images_compare/`** - Gráficos de comparación entre diferentes enfoques
- **`sweep_plots/`** - Visualizaciones de barridos de parámetros
- **`results/`** - Resultados detallados de análisis y reportes

## Tipos de Figuras

### **Comparaciones de Formatos de Precisión**
- Float16 vs Float32 vs Float64 vs Posit16
- Análisis de precisión numérica
- Compromisos entre precisión y eficiencia computacional

### **Análisis de Errores**
- Mediciones de RMSE (Error Cuadrático Medio)
- Error acumulado a lo largo del tiempo
- Análisis logarítmico de errores

### **Comportamiento de Spiking**
- Precisión del timing de spikes
- Análisis de tasas de disparo
- Comparaciones de trenes de spikes

### **Métricas de Calidad**
- PSNR (Relación Señal-Ruido de Pico)
- SSIM (Índice de Similitud Estructural)
- Otras medidas de calidad numérica

## Uso

Cada subdirectorio contiene un archivo README.md explicando las figuras específicas y su propósito. Esta organización facilita encontrar visualizaciones relevantes para diferentes tipos de análisis.

## Convención de Nomenclatura

Las figuras se nombran de manera descriptiva para indicar:
- Qué se está midiendo (ej., `rmse_vs_I`)
- Qué formato de precisión (ej., `_float32`, `_posit16`)
- Qué parámetro varía (ej., `_vs_current`, `_vs_time`) 