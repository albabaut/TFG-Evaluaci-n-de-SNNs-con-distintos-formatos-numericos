# Proyecto de Simulación de Neurociencia con Brian2

Este proyecto implementa y analiza simulaciones de redes neuronales usando Brian2 con diferentes tipos de precisión numérica (float16, float32, float64, posit16).

## Estructura del Proyecto

```
tfg/
├── analysis/                    # Scripts de análisis organizados por categoría
│   ├── neuron_models/          # Simulaciones de neuronas individuales y grupos básicos
│   ├── network_analysis/       # Simulaciones y análisis a nivel de red
│   ├── numerical_precision/    # Comparación de precisión y análisis numérico
│   ├── utils/                  # Funciones de utilidad y scripts auxiliares
│   └── plotting/               # Scripts de graficado y visualización
├── data/                       # Archivos de datos
│   ├── raw/                    # Datos de simulación sin procesar
│   ├── processed/              # Datos procesados
│   └── metrics/                # Métricas y resultados de comparación
├── outputs/                    # Archivos de salida
│   ├── figures/                # Gráficos y visualizaciones generadas
│   │   ├── figuras/            # Figuras principales de análisis
│   │   ├── images/             # Imágenes generales
│   │   ├── images_compare/     # Gráficos de comparación
│   │   ├── sweep_plots/        # Visualizaciones de barridos de parámetros
│   │   └── results/            # Figuras de resultados de análisis
│   ├── reports/                # Reportes de análisis
│   └── logs/                   # Logs de simulación y análisis
├── brian2/                     # Código fuente de la biblioteca Brian2
├── universal/                  # Biblioteca de sistema de números universal
└── venv/                       # Entorno virtual de Python
```

## Categorías de Análisis

### 1. Modelos de Neuronas (`analysis/neuron_models/`)
- Simulaciones de neuronas individuales con diferentes parámetros
- Barridos de parámetros (corriente, constantes de tiempo, umbrales)
- Implementaciones básicas de grupos neuronales
- Modelos de neuronas Izhikevich

### 2. Análisis de Redes (`analysis/network_analysis/`)
- Simulaciones de redes neuronales de spiking
- Implementaciones de aprendizaje STDP
- Análisis de redes multicapa
- Detección de pulsos y análisis de ruido

### 3. Precisión Numérica (`analysis/numerical_precision/`)
- Comparaciones Float16 vs Float32 vs Float64 vs Posit16
- Análisis de errores numéricos
- Casos de prueba de estrés
- Depuración de problemas numéricos

### 4. Utilidades (`analysis/utils/`)
- Funciones auxiliares de análisis
- Cálculo de métricas
- Utilidades de procesamiento de datos
- Scripts de prueba y validación

### 5. Graficado (`analysis/plotting/`)
- Scripts de visualización
- Gráficos de comparación
- Visualizaciones de barridos de parámetros

## Características Principales

- **Soporte multi-preción**: Float16, Float32, Float64 y Posit16
- **Análisis integral**: Desde neuronas individuales hasta redes complejas
- **Barridos de parámetros**: Exploración sistemática de espacios de parámetros
- **Análisis de errores**: RMSE, error acumulado, análisis de timing de spikes
- **Visualización**: Mapas de calor, gráficos de dispersión, series temporales y gráficos estadísticos

## Comenzando

1. **Configurar entorno**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Ejecutar simulaciones básicas**:
   ```bash
   cd analysis/neuron_models
   python neuron1.py
   ```

3. **Ejecutar comparaciones de precisión**:
   ```bash
   cd analysis/numerical_precision
   python comprehensive_numerical_analysis.py
   ```

## Organización de Datos

- **Métricas**: Almacenadas en `data/metrics/` como archivos CSV
- **Figuras**: Generadas en `outputs/figures/` organizadas por tipo de análisis
- **Logs**: Logs de simulación almacenados en `outputs/logs/`
- **Resultados**: Resultados de análisis en `outputs/figures/results/`

## Dependencias

- Brian2 (biblioteca de simulación neural)
- NumPy (computación numérica)
- Matplotlib (graficado)
- Pandas (análisis de datos)
- SciPy (computación científica)
- Wrapper de Posit (implementación personalizada)

## Contribuir

Al añadir nuevos scripts de análisis:
1. Colócalos en el directorio de categoría apropiado
2. Sigue las convenciones de nomenclatura existentes
3. Actualiza este README si añades nuevas categorías
4. Asegúrate de que las salidas vayan a los directorios organizados correctos 
