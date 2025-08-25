
# Análisis Comprehensivo de Formatos Numéricos en Simulaciones Neuronales

**Fecha de análisis:** 2025-06-28 11:53:22

## Resumen Ejecutivo

Este reporte presenta un análisis comprehensivo de las diferencias entre formatos numéricos (float64, float32, float16, posit16) en simulaciones de neuronas LIF usando Brian2.

## Estado de los Análisis


### Step By Step
- **Estado:** completed
- **n_steps:** 500

### Brian2 Callbacks
- **Estado:** failed
- **Error:** 2

### Manual Numpy
- **Estado:** completed
- **n_comparisons:** 500
- **n_divergences:** 2

### Advanced Comparison
- **Estado:** completed
- **n_metrics:** 15
- **n_currents:** 5

### Stress Tests
- **Estado:** failed
- **Error:** Error encountered with object named 'neurongroup'.
Object was created here (most recent call only, full details in debug log):
  File '/home/albab/tfg/stress_test_cases.py', line 173, in test_noise_amplification
    G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',

An error occurred when preparing an object. (See above for original error message and traceback.)

## Conclusiones Principales

### 1. Depuración Paso a Paso
- Se implementaron técnicas de logging detallado durante la simulación
- Se detectaron puntos de divergencia entre formatos numéricos
- Se compararon simulaciones Brian2 con integración manual

### 2. Métricas Avanzadas
- **Jitter de spikes:** Variabilidad en tiempos de disparo
- **Latencia de disparo:** Diferencias en tiempo hasta el primer spike
- **Distancias Victor-Purpura e ISI:** Métricas de similitud entre trenes de spikes
- **Entropía espectral:** Medida de regularidad de patrones de disparo

### 3. Pruebas de Estrés
- **Corriente umbral:** Sensibilidad en el límite de disparo
- **Ruido aleatorio:** Amplificación de diferencias por ruido
- **Simulaciones largas:** Acumulación de errores numéricos
- **Condiciones iniciales aleatorias:** Sensibilidad a condiciones iniciales

## Recomendaciones

1. **Para aplicaciones críticas:** Usar float64 como referencia
2. **Para eficiencia:** Evaluar trade-offs entre precisión y velocidad
3. **Para robustez:** Implementar validaciones numéricas
4. **Para investigación:** Documentar formatos numéricos utilizados

## Archivos Generados

- `comprehensive_analysis.log`: Log detallado del análisis
- `results/comprehensive/`: Directorio con todos los resultados
- Gráficas y tablas de comparación
- Métricas cuantitativas de diferencias

## Metodología

El análisis se basó en las siguientes técnicas:

1. **Depuración numérica paso a paso:** Logging detallado durante la integración
2. **Detección de divergencias:** Identificación de puntos de separación entre trayectorias
3. **Simulación manual:** Control total sobre operaciones numéricas
4. **Métricas avanzadas:** Análisis cuantitativo de diferencias
5. **Pruebas de estrés:** Evaluación en casos límite

## Referencias

- Brian2 Documentation: https://brian2.readthedocs.io/
- Victor-Purpura Distance: Métrica de similitud entre trenes de spikes
- Posit Arithmetic: Formato numérico alternativo a floating-point
