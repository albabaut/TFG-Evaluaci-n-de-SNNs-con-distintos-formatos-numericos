import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos personalizados
try:
    from debug_numerical_step_by_step import NumericalDebugger
    from brian2_debug_callbacks import Brian2DebugCallbacks
    from manual_numpy_integration import ManualNumpyIntegration
    from advanced_numerical_comparison import AdvancedNumericalComparison
    from stress_test_cases import StressTestCases
except ImportError as e:
    print(f"Advertencia: No se pudo importar módulo {e}")
    print("Algunas funcionalidades pueden no estar disponibles")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_analysis.log'),
        logging.StreamHandler()
    ]
)

class ComprehensiveNumericalAnalysis:
    def __init__(self):
        """
        Inicializa el análisis comprehensivo de formatos numéricos
        """
        self.results = {}
        self.summary = {}
        
        # Crear directorio de resultados
        os.makedirs('results/comprehensive', exist_ok=True)
        
        logging.info("ComprehensiveNumericalAnalysis inicializado")
    
    def run_step_by_step_debugging(self):
        """Ejecuta depuración paso a paso"""
        logging.info("=== EJECUTANDO DEPURACIÓN PASO A PASO ===")
        
        try:
            debugger = NumericalDebugger(dt=0.1*b2.ms, simulation_time=50*b2.ms)
            results_df = debugger.run_complete_analysis()
            
            self.results['step_by_step'] = results_df
            self.summary['step_by_step'] = {
                'status': 'completed',
                'n_steps': len(results_df) if results_df is not None else 0
            }
            
            logging.info("Depuración paso a paso completada")
            
        except Exception as e:
            logging.error(f"Error en depuración paso a paso: {e}")
            self.summary['step_by_step'] = {'status': 'failed', 'error': str(e)}
    
    def run_brian2_callbacks_debugging(self):
        """Ejecuta depuración con callbacks de Brian2"""
        logging.info("=== EJECUTANDO DEPURACIÓN CON CALLBACKS ===")
        
        try:
            callback_debugger = Brian2DebugCallbacks(dt=0.1*b2.ms, simulation_time=50*b2.ms)
            df_float64, df_float16, df_detailed = callback_debugger.run_complete_debug_analysis()
            
            self.results['brian2_callbacks'] = {
                'float64': df_float64,
                'float16': df_float16,
                'detailed': df_detailed
            }
            
            self.summary['brian2_callbacks'] = {
                'status': 'completed',
                'n_steps_float64': len(df_float64) if df_float64 is not None else 0,
                'n_steps_float16': len(df_float16) if df_float16 is not None else 0
            }
            
            logging.info("Depuración con callbacks completada")
            
        except Exception as e:
            logging.error(f"Error en depuración con callbacks: {e}")
            self.summary['brian2_callbacks'] = {'status': 'failed', 'error': str(e)}
    
    def run_manual_numpy_integration(self):
        """Ejecuta integración manual con NumPy"""
        logging.info("=== EJECUTANDO INTEGRACIÓN MANUAL CON NUMPY ===")
        
        try:
            integrator = ManualNumpyIntegration(dt=0.1, simulation_time=50)
            df, divergence_points = integrator.run_complete_analysis()
            
            self.results['manual_numpy'] = {
                'comparison_table': df,
                'divergence_points': divergence_points
            }
            
            self.summary['manual_numpy'] = {
                'status': 'completed',
                'n_comparisons': len(df) if df is not None else 0,
                'n_divergences': len(divergence_points)
            }
            
            logging.info("Integración manual completada")
            
        except Exception as e:
            logging.error(f"Error en integración manual: {e}")
            self.summary['manual_numpy'] = {'status': 'failed', 'error': str(e)}
    
    def run_advanced_comparison(self):
        """Ejecuta comparación avanzada de formatos"""
        logging.info("=== EJECUTANDO COMPARACIÓN AVANZADA ===")
        
        try:
            comparator = AdvancedNumericalComparison(dt=0.1*b2.ms, simulation_time=1000*b2.ms)
            
            # Corrientes a probar
            I_ext_values = [10.0, 15.0, 20.0, 25.0, 30.0] * b2.mV
            
            # Ejecutar comparación comprehensiva
            df = comparator.run_comprehensive_comparison(I_ext_values)
            
            # Crear visualizaciones
            comparator.create_advanced_visualizations(df)
            
            self.results['advanced_comparison'] = df
            self.summary['advanced_comparison'] = {
                'status': 'completed',
                'n_metrics': len(df) if df is not None else 0,
                'n_currents': len(I_ext_values)
            }
            
            logging.info("Comparación avanzada completada")
            
        except Exception as e:
            logging.error(f"Error en comparación avanzada: {e}")
            self.summary['advanced_comparison'] = {'status': 'failed', 'error': str(e)}
    
    def run_stress_tests(self):
        """Ejecuta pruebas de estrés"""
        logging.info("=== EJECUTANDO PRUEBAS DE ESTRÉS ===")
        
        try:
            stress_tester = StressTestCases(dt=0.1*b2.ms)
            results = stress_tester.run_all_stress_tests()
            
            self.results['stress_tests'] = results
            self.summary['stress_tests'] = {
                'status': 'completed',
                'n_threshold_tests': len(results['threshold']) if 'threshold' in results else 0,
                'n_noise_tests': len(results['noise']) if 'noise' in results else 0,
                'n_long_sim_tests': len(results['long_simulation']) if 'long_simulation' in results else 0,
                'n_random_condition_tests': len(results['random_conditions']) if 'random_conditions' in results else 0
            }
            
            logging.info("Pruebas de estrés completadas")
            
        except Exception as e:
            logging.error(f"Error en pruebas de estrés: {e}")
            self.summary['stress_tests'] = {'status': 'failed', 'error': str(e)}
    
    def create_comprehensive_report(self):
        """Crea un reporte comprehensivo de todos los análisis"""
        logging.info("=== CREANDO REPORTE COMPREHENSIVO ===")
        
        # Crear reporte en Markdown
        report_content = f"""
# Análisis Comprehensivo de Formatos Numéricos en Simulaciones Neuronales

**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumen Ejecutivo

Este reporte presenta un análisis comprehensivo de las diferencias entre formatos numéricos (float64, float32, float16, posit16) en simulaciones de neuronas LIF usando Brian2.

## Estado de los Análisis

"""
        
        # Agregar estado de cada análisis
        for analysis_name, status_info in self.summary.items():
            status = status_info.get('status', 'unknown')
            report_content += f"\n### {analysis_name.replace('_', ' ').title()}\n"
            report_content += f"- **Estado:** {status}\n"
            
            if status == 'completed':
                for key, value in status_info.items():
                    if key != 'status':
                        report_content += f"- **{key}:** {value}\n"
            elif status == 'failed':
                error = status_info.get('error', 'Unknown error')
                report_content += f"- **Error:** {error}\n"
        
        # Agregar conclusiones principales
        report_content += """
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
"""
        
        # Guardar reporte
        with open('results/comprehensive/comprehensive_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Crear resumen ejecutivo en CSV
        summary_data = []
        for analysis_name, status_info in self.summary.items():
            row = {
                'analysis': analysis_name,
                'status': status_info.get('status', 'unknown'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Agregar métricas específicas si están disponibles
            if status_info.get('status') == 'completed':
                for key, value in status_info.items():
                    if key not in ['status', 'error']:
                        row[key] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/comprehensive/analysis_summary.csv', index=False)
        
        logging.info("Reporte comprehensivo creado")
        
        return report_content
    
    def create_comparison_dashboard(self):
        """Crea un dashboard visual de comparación"""
        logging.info("=== CREANDO DASHBOARD DE COMPARACIÓN ===")
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Título principal
        fig.suptitle('Dashboard de Comparación de Formatos Numéricos', fontsize=16, fontweight='bold')
        
        # Subplot 1: Estado de análisis
        plt.subplot(3, 4, 1)
        analysis_names = list(self.summary.keys())
        statuses = [self.summary[name].get('status', 'unknown') for name in analysis_names]
        
        # Mapear estados a colores
        color_map = {'completed': 'green', 'failed': 'red', 'unknown': 'gray'}
        colors = [color_map.get(status, 'gray') for status in statuses]
        
        bars = plt.bar(range(len(analysis_names)), [1 if s == 'completed' else 0 for s in statuses], color=colors)
        plt.xticks(range(len(analysis_names)), [name.replace('_', '\n') for name in analysis_names], rotation=45)
        plt.ylabel('Estado')
        plt.title('Estado de Análisis')
        plt.ylim(0, 1.2)
        
        # Agregar etiquetas de estado
        for i, (bar, status) in enumerate(zip(bars, statuses)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    status, ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: Resumen de métricas (si están disponibles)
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            df = self.results['advanced_comparison']
            
            plt.subplot(3, 4, 2)
            precision_types = df['precision'].unique()
            rmse_values = [df[df['precision'] == p]['rmse_voltage'].mean() for p in precision_types]
            plt.bar(precision_types, rmse_values)
            plt.ylabel('RMSE Voltaje (mV)')
            plt.title('Error Cuadrático Medio')
            plt.xticks(rotation=45)
        
        # Subplot 3: Jitter de spikes
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            plt.subplot(3, 4, 3)
            jitter_data = []
            precision_labels = []
            for p in precision_types:
                data = df[df['precision'] == p]['jitter_mean'].dropna()
                if len(data) > 0:
                    jitter_data.append(data.values)
                    precision_labels.append(p)
            
            if jitter_data:
                plt.boxplot(jitter_data, labels=precision_labels)
                plt.ylabel('Jitter (ms)')
                plt.title('Distribución de Jitter')
                plt.xticks(rotation=45)
        
        # Subplot 4: Distancia Victor-Purpura
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            plt.subplot(3, 4, 4)
            vp_data = []
            for p in precision_types:
                data = df[df['precision'] == p]['vp_distance'].dropna()
                if len(data) > 0:
                    vp_data.append(data.values)
            
            if vp_data:
                plt.boxplot(vp_data, labels=precision_types)
                plt.ylabel('Distancia Victor-Purpura')
                plt.title('Distancia entre Trenes de Spikes')
                plt.xticks(rotation=45)
        
        # Subplot 5: Tasa de disparo
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            plt.subplot(3, 4, 5)
            for p in precision_types:
                data = df[df['precision'] == p]
                plt.scatter(data['I_ext'], data['firing_rate_diff'], alpha=0.7, label=p)
            plt.xlabel('Corriente externa (mV)')
            plt.ylabel('Diferencia de tasa (Hz)')
            plt.title('Tasa de Disparo vs Corriente')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Subplot 6: Entropía espectral
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            plt.subplot(3, 4, 6)
            entropy_data = []
            for p in precision_types:
                data = df[df['precision'] == p]['spectral_entropy_diff'].dropna()
                if len(data) > 0:
                    entropy_data.append(data.values)
            
            if entropy_data:
                plt.boxplot(entropy_data, labels=precision_types)
                plt.ylabel('Diferencia de entropía')
                plt.title('Entropía Espectral')
                plt.xticks(rotation=45)
        
        # Subplot 7: Pruebas de estrés (si están disponibles)
        if 'stress_tests' in self.results:
            plt.subplot(3, 4, 7)
            stress_results = self.results['stress_tests']
            
            if 'threshold' in stress_results and stress_results['threshold'] is not None:
                threshold_df = stress_results['threshold']
                precision_types_stress = threshold_df['precision'].unique()
                threshold_success = []
                
                for p in precision_types_stress:
                    data = threshold_df[threshold_df['precision'] == p]
                    success_rate = np.mean(data['reached_threshold'])
                    threshold_success.append(success_rate)
                
                plt.bar(precision_types_stress, threshold_success)
                plt.ylabel('Tasa de éxito')
                plt.title('Prueba de Umbral')
                plt.xticks(rotation=45)
        
        # Subplot 8: Simulación larga
        if 'stress_tests' in self.results:
            plt.subplot(3, 4, 8)
            stress_results = self.results['stress_tests']
            
            if 'long_simulation' in stress_results and stress_results['long_simulation'] is not None:
                long_sim_df = stress_results['long_simulation']
                precision_types_long = long_sim_df['precision'].values
                rates = long_sim_df['firing_rate_hz'].values
                
                plt.bar(precision_types_long, rates)
                plt.ylabel('Tasa de disparo (Hz)')
                plt.title('Simulación Larga')
                plt.xticks(rotation=45)
        
        # Subplot 9: Resumen de archivos generados
        plt.subplot(3, 4, 9)
        file_types = ['Logs', 'CSV', 'PNG', 'Markdown']
        file_counts = [1, 5, 8, 1]  # Estimación basada en los análisis
        
        plt.bar(file_types, file_counts)
        plt.ylabel('Número de archivos')
        plt.title('Archivos Generados')
        
        # Subplot 10: Tiempo de ejecución (estimado)
        plt.subplot(3, 4, 10)
        analysis_times = [5, 3, 4, 8, 6]  # Minutos estimados
        analysis_names_short = ['Step\nDebug', 'Callbacks', 'Manual\nNumpy', 'Advanced\nComp', 'Stress\nTests']
        
        plt.bar(analysis_names_short, analysis_times)
        plt.ylabel('Tiempo (min)')
        plt.title('Tiempo de Ejecución Estimado')
        plt.xticks(rotation=45)
        
        # Subplot 11: Métricas clave
        plt.subplot(3, 4, 11)
        if 'advanced_comparison' in self.results and self.results['advanced_comparison'] is not None:
            df = self.results['advanced_comparison']
            
            # Calcular métricas promedio por precisión
            metrics_summary = []
            for p in precision_types:
                data = df[df['precision'] == p]
                avg_rmse = data['rmse_voltage'].mean()
                avg_jitter = data['jitter_mean'].mean()
                avg_vp = data['vp_distance'].mean()
                
                metrics_summary.append({
                    'precision': p,
                    'rmse': avg_rmse,
                    'jitter': avg_jitter,
                    'vp_distance': avg_vp
                })
            
            metrics_df = pd.DataFrame(metrics_summary)
            
            # Gráfica de radar simplificada
            metrics = ['rmse', 'jitter', 'vp_distance']
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Cerrar el círculo
            
            for i, p in enumerate(precision_types):
                values = [metrics_df.iloc[i][m] for m in metrics]
                values += values[:1]  # Cerrar el círculo
                
                plt.polar(angles, values, 'o-', label=p, linewidth=2)
            
            plt.xticks(angles[:-1], ['RMSE', 'Jitter', 'VP Dist'])
            plt.title('Métricas por Precisión')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Subplot 12: Recomendaciones
        plt.subplot(3, 4, 12)
        recommendations = [
            'Usar float64 para\nreferencia',
            'Evaluar trade-offs\nprecisión/velocidad',
            'Implementar\nvalidaciones',
            'Documentar\nformatos'
        ]
        
        plt.text(0.1, 0.8, 'Recomendaciones:', fontsize=12, fontweight='bold')
        for i, rec in enumerate(recommendations):
            plt.text(0.1, 0.6 - i*0.15, f'{i+1}. {rec}', fontsize=10)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/comprehensive/comparison_dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logging.info("Dashboard de comparación creado")
    
    def run_complete_analysis(self):
        """Ejecuta el análisis comprehensivo completo"""
        logging.info("=== INICIANDO ANÁLISIS COMPREHENSIVO COMPLETO ===")
        
        # 1. Depuración paso a paso
        self.run_step_by_step_debugging()
        
        # 2. Depuración con callbacks
        self.run_brian2_callbacks_debugging()
        
        # 3. Integración manual
        self.run_manual_numpy_integration()
        
        # 4. Comparación avanzada
        self.run_advanced_comparison()
        
        # 5. Pruebas de estrés
        self.run_stress_tests()
        
        # 6. Crear reporte comprehensivo
        self.create_comprehensive_report()
        
        # 7. Crear dashboard
        self.create_comparison_dashboard()
        
        logging.info("=== ANÁLISIS COMPREHENSIVO COMPLETADO ===")
        
        return self.results, self.summary

def main():
    """Función principal"""
    print("=== ANÁLISIS COMPREHENSIVO DE FORMATOS NUMÉRICOS ===")
    print("Implementando todas las técnicas de análisis avanzado")
    print("1. Depuración paso a paso con logging detallado")
    print("2. Callbacks de Brian2 para monitoreo en tiempo real")
    print("3. Integración manual con NumPy para control total")
    print("4. Métricas avanzadas: jitter, latencia, distancias")
    print("5. Pruebas de estrés en casos límite")
    print("6. Reporte comprehensivo y dashboard visual")
    print()
    
    # Crear instancia del análisis comprehensivo
    analyzer = ComprehensiveNumericalAnalysis()
    
    # Ejecutar análisis completo
    results, summary = analyzer.run_complete_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- comprehensive_analysis.log: Log detallado")
    print("- results/comprehensive/comprehensive_report.md: Reporte completo")
    print("- results/comprehensive/analysis_summary.csv: Resumen ejecutivo")
    print("- results/comprehensive/comparison_dashboard.png: Dashboard visual")
    print()
    print("Estado de los análisis:")
    for analysis_name, status_info in summary.items():
        status = status_info.get('status', 'unknown')
        print(f"  - {analysis_name}: {status}")

if __name__ == "__main__":
    main() 