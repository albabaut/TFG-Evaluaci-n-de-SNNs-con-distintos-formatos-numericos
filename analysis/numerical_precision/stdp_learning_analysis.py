import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from posit_wrapper import convert16
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stdp_learning_analysis.log'),
        logging.StreamHandler()
    ]
)

class STDPLearningAnalysis:
    def __init__(self, dt=0.1*b2.ms, simulation_time=5000*b2.ms):
        """
        Inicializa el análisis de aprendizaje con STDP
        
        Args:
            dt: Paso de tiempo
            simulation_time: Tiempo de simulación
        """
        self.dt = dt
        self.simulation_time = simulation_time
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        
        # Parámetros STDP
        self.tau_pre = 20 * b2.ms
        self.tau_post = 20 * b2.ms
        self.A_pre = 0.01
        self.A_post = -0.005
        self.w_max = 0.1
        
        # Configurar Brian2
        b2.prefs.codegen.target = 'numpy'
        
        logging.info(f"STDPLearningAnalysis inicializado")
    
    def create_stdp_network(self, n_neurons=10, precision_type='float64', input_pattern=None):
        """
        Crea una red con aprendizaje STDP
        
        Args:
            n_neurons: Número de neuronas
            precision_type: Tipo de precisión numérica
            input_pattern: Patrón de entrada (opcional)
        
        Returns:
            network_components: Diccionario con componentes de la red
        """
        logging.info(f"Creando red STDP: {n_neurons} neuronas con precisión {precision_type}")
        
        # Configurar dtype según precisión
        if precision_type == 'float64':
            dtype = np.float64
        elif precision_type == 'float32':
            dtype = np.float32
        elif precision_type == 'float16':
            dtype = np.float16
        elif precision_type == 'posit16':
            dtype = np.float32  # Base, luego convertir
        
        # Ecuaciones de la neurona
        eqs = '''
        dv/dt = (-(V_rest - v) + I_syn) / tau : volt (unless refractory)
        I_syn : volt
        '''
        
        # Crear grupo neuronal
        neurons = b2.NeuronGroup(n_neurons, eqs, threshold='v > Vth', reset='v = Vreset',
                                refractory=5*b2.ms, method='euler', dtype=dtype)
        neurons.v = self.V_rest
        
        # Aplicar conversión de precisión si es necesario
        if precision_type == 'posit16':
            neurons.v = np.array([convert16(float(v / b2.mV)) * b2.mV for v in neurons.v])
        
        # Crear sinapsis con STDP
        stdp_eqs = '''
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        '''
        
        stdp_pre = '''
        v_post += w*mV
        apre += Apre
        w = clip(w + apost, 0, wmax)
        '''
        
        stdp_post = '''
        apost += Apost
        w = clip(w + apre, 0, wmax)
        '''
        
        synapses = b2.Synapses(neurons, neurons, 
                              model=stdp_eqs,
                              on_pre=stdp_pre,
                              on_post=stdp_post,
                              dtype=dtype)
        
        # Conectar todas las neuronas entre sí (excepto autoconexiones)
        synapses.connect(condition='i != j')
        
        # Inicializar pesos aleatoriamente
        synapses.w = np.random.uniform(0, 0.05, len(synapses))
        
        # Aplicar conversión de precisión a pesos si es necesario
        if precision_type == 'posit16':
            synapses.w = np.array([convert16(w) for w in synapses.w])
        
        # Crear input si se especifica
        if input_pattern is not None:
            input_neurons = b2.NeuronGroup(len(input_pattern), 
                                         'I_ext : volt',
                                         dtype=dtype)
            input_neurons.I_ext = input_pattern
            
            # Conectar input a neuronas
            input_synapses = b2.Synapses(input_neurons, neurons,
                                       on_pre='I_syn_post += I_ext_pre',
                                       dtype=dtype)
            input_synapses.connect(j='i')  # Conexión uno a uno
        else:
            input_neurons = None
            input_synapses = None
        
        # Monitores
        voltage_monitor = b2.StateMonitor(neurons, 'v', record=True)
        spike_monitor = b2.SpikeMonitor(neurons)
        weight_monitor = b2.StateMonitor(synapses, 'w', record=True)
        
        network_components = {
            'neurons': neurons,
            'synapses': synapses,
            'input_neurons': input_neurons,
            'input_synapses': input_synapses,
            'voltage_monitor': voltage_monitor,
            'spike_monitor': spike_monitor,
            'weight_monitor': weight_monitor,
            'precision': precision_type
        }
        
        return network_components
    
    def run_stdp_simulation(self, n_neurons=10, input_pattern=None, precision_types=['float64', 'float32', 'float16', 'posit16']):
        """
        Ejecuta simulación STDP con diferentes precisiones
        
        Args:
            n_neurons: Número de neuronas
            input_pattern: Patrón de entrada
            precision_types: Tipos de precisión a probar
        
        Returns:
            results: Diccionario con resultados por precisión
        """
        logging.info("=== EJECUTANDO SIMULACIÓN STDP ===")
        
        results = {}
        
        for precision in precision_types:
            logging.info(f"Ejecutando STDP con precisión: {precision}")
            
            # Limpiar estado de Brian2
            b2.device.reinit()
            b2.device.activate()
            
            # Crear red
            network = self.create_stdp_network(n_neurons, precision, input_pattern)
            
            # Crear Network de Brian2
            all_objects = [network['neurons'], network['synapses']]
            if network['input_neurons'] is not None:
                all_objects.extend([network['input_neurons'], network['input_synapses']])
            all_objects.extend([network['voltage_monitor'], network['spike_monitor'], network['weight_monitor']])
            
            net = b2.Network(all_objects)
            
            # Ejecutar simulación
            namespace = {
                'tau': self.tau, 'V_rest': self.V_rest, 
                'Vth': self.Vth, 'Vreset': self.Vreset,
                'taupre': self.tau_pre, 'taupost': self.tau_post,
                'Apre': self.A_pre, 'Apost': self.A_post,
                'wmax': self.w_max
            }
            
            net.run(self.simulation_time, namespace=namespace)
            
            # Extraer resultados
            voltage = network['voltage_monitor'].v / b2.mV
            time = network['voltage_monitor'].t / b2.ms
            spikes = network['spike_monitor'].t / b2.ms
            spike_neurons = network['spike_monitor'].i
            weights = network['weight_monitor'].w
            weight_time = network['weight_monitor'].t / b2.ms
            
            # Calcular métricas de aprendizaje
            initial_weights = weights[:, 0]
            final_weights = weights[:, -1]
            weight_changes = final_weights - initial_weights
            
            # Calcular métricas de actividad
            n_spikes_per_neuron = np.bincount(spike_neurons, minlength=n_neurons)
            firing_rates = n_spikes_per_neuron / (float(self.simulation_time / b2.second))
            
            # Calcular correlaciones de peso
            weight_correlations = []
            for i in range(len(weights)):
                if np.std(weights[i]) > 0:
                    correlation = np.corrcoef(weight_time, weights[i])[0, 1]
                    weight_correlations.append(correlation)
            
            results[precision] = {
                'voltage': voltage,
                'time': time,
                'spikes': spikes,
                'spike_neurons': spike_neurons,
                'weights': weights,
                'weight_time': weight_time,
                'initial_weights': initial_weights,
                'final_weights': final_weights,
                'weight_changes': weight_changes,
                'firing_rates': firing_rates,
                'weight_correlations': weight_correlations,
                'network': network
            }
            
            logging.info(f"  {precision}: {len(spikes)} spikes, {len(weights)} sinapsis")
        
        return results
    
    def analyze_learning_differences(self, results):
        """
        Analiza las diferencias en el aprendizaje entre precisiones
        
        Args:
            results: Resultados de las simulaciones
        
        Returns:
            analysis: Análisis de diferencias de aprendizaje
        """
        logging.info("=== ANALIZANDO DIFERENCIAS DE APRENDIZAJE ===")
        
        # Usar float64 como referencia
        reference = results['float64']
        analysis = {}
        
        for precision in ['float32', 'float16', 'posit16']:
            if precision not in results:
                continue
            
            test = results[precision]
            precision_analysis = {}
            
            # Comparar pesos finales
            ref_final_weights = reference['final_weights']
            test_final_weights = test['final_weights']
            
            weight_rmse = np.sqrt(np.mean((ref_final_weights - test_final_weights) ** 2))
            weight_correlation = np.corrcoef(ref_final_weights, test_final_weights)[0, 1]
            
            # Comparar cambios de peso
            ref_weight_changes = reference['weight_changes']
            test_weight_changes = test['weight_changes']
            
            change_rmse = np.sqrt(np.mean((ref_weight_changes - test_weight_changes) ** 2))
            change_correlation = np.corrcoef(ref_weight_changes, test_weight_changes)[0, 1]
            
            # Comparar tasas de disparo
            ref_firing_rates = reference['firing_rates']
            test_firing_rates = test['firing_rates']
            
            rate_rmse = np.sqrt(np.mean((ref_firing_rates - test_firing_rates) ** 2))
            rate_correlation = np.corrcoef(ref_firing_rates, test_firing_rates)[0, 1]
            
            # Comparar número total de spikes
            ref_n_spikes = len(reference['spikes'])
            test_n_spikes = len(test['spikes'])
            spike_diff = test_n_spikes - ref_n_spikes
            
            # Análisis de estabilidad de pesos
            ref_weight_std = np.std(reference['weights'], axis=1)
            test_weight_std = np.std(test['weights'], axis=1)
            
            stability_correlation = np.corrcoef(ref_weight_std, test_weight_std)[0, 1]
            
            precision_analysis = {
                'weight_rmse': weight_rmse,
                'weight_correlation': weight_correlation,
                'change_rmse': change_rmse,
                'change_correlation': change_correlation,
                'rate_rmse': rate_rmse,
                'rate_correlation': rate_correlation,
                'spike_diff': spike_diff,
                'stability_correlation': stability_correlation,
                'ref_n_spikes': ref_n_spikes,
                'test_n_spikes': test_n_spikes
            }
            
            analysis[precision] = precision_analysis
        
        return analysis
    
    def create_stdp_visualizations(self, results, analysis):
        """
        Crea visualizaciones para el análisis STDP
        
        Args:
            results: Resultados de las simulaciones
            analysis: Análisis de diferencias de aprendizaje
        """
        logging.info("=== CREANDO VISUALIZACIONES STDP ===")
        
        os.makedirs('results/stdp_analysis', exist_ok=True)
        
        # 1. Evolución de pesos en el tiempo
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mostrar evolución de algunos pesos representativos
        n_weights_to_show = min(5, len(results['float64']['weights']))
        
        for i in range(n_weights_to_show):
            for j, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
                if precision in results:
                    ax = axes[i//3, i%3] if i < 3 else axes[1, i-3]
                    
                    weight_time = results[precision]['weight_time']
                    weights = results[precision]['weights'][i]
                    
                    ax.plot(weight_time, weights, label=precision, alpha=0.8)
                    ax.set_xlabel('Tiempo (ms)')
                    ax.set_ylabel('Peso sináptico')
                    ax.set_title(f'Evolución del Peso {i+1}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stdp_analysis/weight_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Distribución de pesos finales
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision in results:
                ax = axes[i//2, i%2]
                
                final_weights = results[precision]['final_weights']
                initial_weights = results[precision]['initial_weights']
                
                ax.hist(initial_weights, alpha=0.7, label='Inicial', bins=20)
                ax.hist(final_weights, alpha=0.7, label='Final', bins=20)
                ax.set_xlabel('Peso sináptico')
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'Distribución de Pesos - {precision}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stdp_analysis/weight_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Comparación de cambios de peso
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Subplot 1: Cambios de peso vs precisión
        ax = axes[0, 0]
        for precision in ['float32', 'float16', 'posit16']:
            if precision in results:
                weight_changes = results[precision]['weight_changes']
                ax.hist(weight_changes, alpha=0.7, label=precision, bins=20)
        ax.set_xlabel('Cambio de peso')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Cambios de Peso')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Tasas de disparo
        ax = axes[0, 1]
        for precision in ['float64', 'float32', 'float16', 'posit16']:
            if precision in results:
                firing_rates = results[precision]['firing_rates']
                ax.hist(firing_rates, alpha=0.7, label=precision, bins=20)
        ax.set_xlabel('Tasa de disparo (Hz)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Tasas de Disparo')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: Raster plot de spikes
        ax = axes[1, 0]
        for precision in ['float64', 'float32', 'float16', 'posit16']:
            if precision in results:
                spikes = results[precision]['spikes']
                spike_neurons = results[precision]['spike_neurons']
                ax.scatter(spikes, spike_neurons, s=1, alpha=0.7, label=precision)
        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Neurona')
        ax.set_title('Raster Plot de Spikes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 4: Métricas de comparación
        ax = axes[1, 1]
        if analysis:
            metrics = ['weight_rmse', 'change_rmse', 'rate_rmse']
            metric_names = ['RMSE Pesos', 'RMSE Cambios', 'RMSE Tasas']
            
            x = np.arange(len(metrics))
            width = 0.25
            
            for i, precision in enumerate(['float32', 'float16', 'posit16']):
                if precision in analysis:
                    values = [analysis[precision][m] for m in metrics]
                    ax.bar(x + i*width, values, width, label=precision, alpha=0.8)
            
            ax.set_xlabel('Métricas')
            ax.set_ylabel('RMSE')
            ax.set_title('Comparación de Métricas')
            ax.set_xticks(x + width)
            ax.set_xticklabels(metric_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/stdp_analysis/learning_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Heatmap de correlaciones de peso
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, precision in enumerate(['float32', 'float16', 'posit16']):
            if precision in results:
                ax = axes[i]
                
                # Crear matriz de correlaciones entre pesos
                weights = results[precision]['weights']
                n_weights = len(weights)
                
                if n_weights > 1:
                    correlation_matrix = np.zeros((n_weights, n_weights))
                    
                    for j in range(n_weights):
                        for k in range(n_weights):
                            if j == k:
                                correlation_matrix[j, k] = 1.0
                            else:
                                correlation = np.corrcoef(weights[j], weights[k])[0, 1]
                                correlation_matrix[j, k] = correlation if not np.isnan(correlation) else 0.0
                    
                    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title(f'Correlaciones de Pesos - {precision}')
                    ax.set_xlabel('Peso')
                    ax.set_ylabel('Peso')
        
        plt.tight_layout()
        plt.savefig('results/stdp_analysis/weight_correlations.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_learning_report(self, results, analysis):
        """
        Crea reporte detallado del aprendizaje STDP
        
        Args:
            results: Resultados de las simulaciones
            analysis: Análisis de diferencias de aprendizaje
        """
        logging.info("=== CREANDO REPORTE DE APRENDIZAJE STDP ===")
        
        # Crear DataFrame con todos los datos
        report_data = []
        
        for precision in results.keys():
            result = results[precision]
            
            for i in range(len(result['weights'])):
                row = {
                    'precision': precision,
                    'synapse_id': i,
                    'initial_weight': result['initial_weights'][i],
                    'final_weight': result['final_weights'][i],
                    'weight_change': result['weight_changes'][i],
                    'weight_std': np.std(result['weights'][i]),
                    'weight_correlation': result['weight_correlations'][i] if i < len(result['weight_correlations']) else np.nan
                }
                report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Agregar métricas de comparación
        comparison_data = []
        for precision in analysis.keys():
            metrics = analysis[precision]
            row = {
                'precision': precision,
                'weight_rmse': metrics['weight_rmse'],
                'weight_correlation': metrics['weight_correlation'],
                'change_rmse': metrics['change_rmse'],
                'change_correlation': metrics['change_correlation'],
                'rate_rmse': metrics['rate_rmse'],
                'rate_correlation': metrics['rate_correlation'],
                'spike_diff': metrics['spike_diff'],
                'stability_correlation': metrics['stability_correlation']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Guardar reportes
        df.to_csv('results/stdp_analysis/learning_detailed_report.csv', index=False)
        comparison_df.to_csv('results/stdp_analysis/learning_comparison_report.csv', index=False)
        
        # Imprimir resumen
        print("\n=== RESUMEN DE APRENDIZAJE STDP ===")
        for precision in analysis.keys():
            metrics = analysis[precision]
            print(f"\n{precision}:")
            print(f"  RMSE Pesos: {metrics['weight_rmse']:.6f}")
            print(f"  Correlación Pesos: {metrics['weight_correlation']:.3f}")
            print(f"  RMSE Cambios: {metrics['change_rmse']:.6f}")
            print(f"  Correlación Cambios: {metrics['change_correlation']:.3f}")
            print(f"  RMSE Tasas: {metrics['rate_rmse']:.3f}")
            print(f"  Diferencia Spikes: {metrics['spike_diff']}")
        
        return df, comparison_df
    
    def run_comprehensive_stdp_analysis(self):
        """
        Ejecuta análisis comprehensivo de aprendizaje STDP
        """
        logging.info("=== INICIANDO ANÁLISIS COMPREHENSIVO STDP ===")
        
        # Configuraciones a probar
        configs = [
            {'name': 'small', 'n_neurons': 10, 'input_pattern': None},
            {'name': 'medium', 'n_neurons': 20, 'input_pattern': None},
            {'name': 'with_input', 'n_neurons': 15, 'input_pattern': [10.0, 15.0, 20.0] * b2.mV}
        ]
        
        all_results = {}
        
        for config in configs:
            logging.info(f"Probando configuración: {config['name']}")
            
            # Ejecutar simulación
            results = self.run_stdp_simulation(
                n_neurons=config['n_neurons'],
                input_pattern=config['input_pattern']
            )
            
            # Analizar diferencias
            analysis = self.analyze_learning_differences(results)
            
            # Crear visualizaciones
            self.create_stdp_visualizations(results, analysis)
            
            # Crear reporte
            df, comparison_df = self.create_learning_report(results, analysis)
            
            # Almacenar resultados
            all_results[config['name']] = {
                'config': config,
                'results': results,
                'analysis': analysis,
                'detailed_report': df,
                'comparison_report': comparison_df
            }
        
        logging.info("=== ANÁLISIS STDP COMPLETADO ===")
        
        return all_results

def main():
    """Función principal"""
    print("=== ANÁLISIS DE APRENDIZAJE STDP ===")
    print("Estudiando impacto de precisión numérica en aprendizaje sináptico")
    print("1. Redes con plasticidad STDP")
    print("2. Análisis de evolución de pesos")
    print("3. Comparación de patrones de aprendizaje")
    print("4. Reporte de diferencias de aprendizaje")
    print()
    
    # Crear instancia del analizador
    analyzer = STDPLearningAnalysis(dt=0.1*b2.ms, simulation_time=5000*b2.ms)
    
    # Ejecutar análisis comprehensivo
    results = analyzer.run_comprehensive_stdp_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- stdp_learning_analysis.log: Log detallado")
    print("- results/stdp_analysis/weight_evolution.png: Evolución de pesos")
    print("- results/stdp_analysis/weight_distributions.png: Distribuciones de pesos")
    print("- results/stdp_analysis/learning_comparison.png: Comparación de aprendizaje")
    print("- results/stdp_analysis/learning_detailed_report.csv: Reporte detallado")
    print("- results/stdp_analysis/learning_comparison_report.csv: Reporte de comparación")

if __name__ == "__main__":
    main() 