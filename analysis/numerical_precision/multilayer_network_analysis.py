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
        logging.FileHandler('multilayer_network_analysis.log'),
        logging.StreamHandler()
    ]
)

class MultilayerNetworkAnalysis:
    def __init__(self, dt=0.1*b2.ms, simulation_time=1000*b2.ms):
        """
        Inicializa el análisis de redes multicapa
        
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
        
        # Configurar Brian2
        b2.prefs.codegen.target = 'numpy'
        
        logging.info(f"MultilayerNetworkAnalysis inicializado")
    
    def create_multilayer_network(self, layer_sizes, precision_type='float64', input_current=None):
        """
        Crea una red feedforward multicapa
        
        Args:
            layer_sizes: Lista con número de neuronas por capa
            precision_type: Tipo de precisión numérica
            input_current: Corriente de entrada para la primera capa
        
        Returns:
            network_components: Diccionario con componentes de la red
        """
        logging.info(f"Creando red multicapa: {layer_sizes} con precisión {precision_type}")
        
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
        
        # Crear capas
        layers = []
        monitors = []
        spike_monitors = []
        
        for i, n_neurons in enumerate(layer_sizes):
            # Crear grupo neuronal
            layer = b2.NeuronGroup(n_neurons, eqs, threshold='v > Vth', reset='v = Vreset',
                                  refractory=5*b2.ms, method='euler', dtype=dtype)
            
            # Condiciones iniciales
            layer.v = self.V_rest
            
            # Aplicar conversión de precisión si es necesario
            if precision_type == 'posit16':
                layer.v = np.array([convert16(float(v / b2.mV)) * b2.mV for v in layer.v])
            
            layers.append(layer)
            
            # Monitores
            mon = b2.StateMonitor(layer, 'v', record=True)
            spk = b2.SpikeMonitor(layer)
            
            monitors.append(mon)
            spike_monitors.append(spk)
        
        # Crear sinapsis entre capas
        synapses = []
        for i in range(len(layers) - 1):
            # Sinapsis excitatorias con peso fijo
            syn = b2.Synapses(layers[i], layers[i+1], 
                             on_pre='I_syn_post += 5*mV',
                             dtype=dtype)
            
            # Conectar todas las neuronas de la capa i con todas las de la capa i+1
            syn.connect()
            synapses.append(syn)
        
        # Crear input para la primera capa
        if input_current is not None:
            input_layer = b2.NeuronGroup(layer_sizes[0], 
                                       'I_ext : volt',
                                       dtype=dtype)
            input_layer.I_ext = input_current
            
            # Conectar input a primera capa
            input_syn = b2.Synapses(input_layer, layers[0],
                                   on_pre='I_syn_post += I_ext_pre',
                                   dtype=dtype)
            input_syn.connect(j='i')  # Conexión uno a uno
            
            synapses.insert(0, input_syn)
            layers.insert(0, input_layer)
            monitors.insert(0, None)
            spike_monitors.insert(0, None)
        
        # Crear Network
        network_components = {
            'layers': layers,
            'synapses': synapses,
            'monitors': monitors,
            'spike_monitors': spike_monitors,
            'precision': precision_type
        }
        
        return network_components
    
    def run_multilayer_simulation(self, layer_sizes, input_current, precision_types=['float64', 'float32', 'float16', 'posit16']):
        """
        Ejecuta simulación de red multicapa con diferentes precisiones
        
        Args:
            layer_sizes: Lista con número de neuronas por capa
            input_current: Corriente de entrada
            precision_types: Tipos de precisión a probar
        
        Returns:
            results: Diccionario con resultados por precisión
        """
        logging.info("=== EJECUTANDO SIMULACIÓN DE RED MULTICAPA ===")
        
        results = {}
        
        for precision in precision_types:
            logging.info(f"Ejecutando con precisión: {precision}")
            
            # Limpiar estado de Brian2
            b2.device.reinit()
            b2.device.activate()
            
            # Crear red
            network = self.create_multilayer_network(layer_sizes, precision, input_current)
            
            # Crear Network de Brian2
            all_objects = []
            for layer in network['layers']:
                all_objects.append(layer)
            for syn in network['synapses']:
                all_objects.append(syn)
            for mon in network['monitors']:
                if mon is not None:
                    all_objects.append(mon)
            for spk in network['spike_monitors']:
                if spk is not None:
                    all_objects.append(spk)
            
            net = b2.Network(all_objects)
            
            # Ejecutar simulación
            namespace = {
                'tau': self.tau, 'V_rest': self.V_rest, 
                'Vth': self.Vth, 'Vreset': self.Vreset
            }
            
            net.run(self.simulation_time, namespace=namespace)
            
            # Extraer resultados
            layer_results = []
            for i, (mon, spk) in enumerate(zip(network['monitors'], network['spike_monitors'])):
                if mon is not None and spk is not None:
                    layer_data = {
                        'layer': i,
                        'voltage': mon.v / b2.mV,
                        'time': mon.t / b2.ms,
                        'spikes': spk.t / b2.ms,
                        'spike_neurons': spk.i,
                        'n_neurons': len(mon.v)
                    }
                    layer_results.append(layer_data)
            
            results[precision] = {
                'layers': layer_results,
                'network': network
            }
            
            logging.info(f"  {precision}: {len(layer_results)} capas simuladas")
        
        return results
    
    def analyze_error_propagation(self, results):
        """
        Analiza la propagación de errores de capa a capa
        
        Args:
            results: Resultados de las simulaciones
        
        Returns:
            analysis: Análisis de propagación de errores
        """
        logging.info("=== ANALIZANDO PROPAGACIÓN DE ERRORES ===")
        
        # Usar float64 como referencia
        reference = results['float64']
        analysis = {}
        
        for precision in ['float32', 'float16', 'posit16']:
            if precision not in results:
                continue
            
            test = results[precision]
            precision_analysis = {}
            
            for layer_idx in range(min(len(reference['layers']), len(test['layers']))):
                ref_layer = reference['layers'][layer_idx]
                test_layer = test['layers'][layer_idx]
                
                # Métricas de voltaje
                ref_voltage = ref_layer['voltage']
                test_voltage = test_layer['voltage']
                
                # Calcular errores
                rmse_voltage = np.sqrt(np.mean((ref_voltage - test_voltage) ** 2))
                max_error = np.max(np.abs(ref_voltage - test_voltage))
                
                # Métricas de spikes
                ref_spikes = ref_layer['spikes']
                test_spikes = test_layer['spikes']
                
                ref_n_spikes = len(ref_spikes)
                test_n_spikes = len(test_spikes)
                spike_diff = test_n_spikes - ref_n_spikes
                
                # Tasa de disparo
                ref_rate = ref_n_spikes / (float(self.simulation_time / b2.second))
                test_rate = test_n_spikes / (float(self.simulation_time / b2.second))
                rate_diff = test_rate - ref_rate
                
                # Correlación de voltajes
                if ref_voltage.shape == test_voltage.shape:
                    correlation = np.corrcoef(ref_voltage.flatten(), test_voltage.flatten())[0, 1]
                else:
                    correlation = np.nan
                
                layer_metrics = {
                    'rmse_voltage': rmse_voltage,
                    'max_error': max_error,
                    'spike_diff': spike_diff,
                    'rate_diff': rate_diff,
                    'correlation': correlation,
                    'ref_n_spikes': ref_n_spikes,
                    'test_n_spikes': test_n_spikes,
                    'ref_rate': ref_rate,
                    'test_rate': test_rate
                }
                
                precision_analysis[f'layer_{layer_idx}'] = layer_metrics
            
            analysis[precision] = precision_analysis
        
        return analysis
    
    def create_multilayer_visualizations(self, results, analysis):
        """
        Crea visualizaciones para el análisis de redes multicapa
        
        Args:
            results: Resultados de las simulaciones
            analysis: Análisis de propagación de errores
        """
        logging.info("=== CREANDO VISUALIZACIONES MULTICAPA ===")
        
        os.makedirs('results/multilayer_analysis', exist_ok=True)
        
        # 1. Raster plots por capa y precisión
        fig, axes = plt.subplots(len(results), len(results['float64']['layers']), 
                                figsize=(15, 12))
        
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        if len(results['float64']['layers']) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            if precision not in results:
                continue
            
            for j, layer_data in enumerate(results[precision]['layers']):
                ax = axes[i, j]
                
                # Crear raster plot
                spikes = layer_data['spikes']
                spike_neurons = layer_data['spike_neurons']
                
                if len(spikes) > 0:
                    ax.scatter(spikes, spike_neurons, s=1, alpha=0.7)
                
                ax.set_xlabel('Tiempo (ms)')
                ax.set_ylabel('Neurona')
                ax.set_title(f'{precision} - Capa {j}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/multilayer_analysis/raster_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Propagación de errores por capa
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['rmse_voltage', 'spike_diff', 'rate_diff', 'correlation']
        metric_names = ['RMSE Voltaje', 'Diferencia de Spikes', 'Diferencia de Tasa', 'Correlación']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            for precision in ['float32', 'float16', 'posit16']:
                if precision not in analysis:
                    continue
                
                layers = []
                values = []
                
                for layer_key in analysis[precision].keys():
                    layer_num = int(layer_key.split('_')[1])
                    layers.append(layer_num)
                    values.append(analysis[precision][layer_key][metric])
                
                ax.plot(layers, values, 'o-', label=precision, linewidth=2)
            
            ax.set_xlabel('Capa')
            ax.set_ylabel(name)
            ax.set_title(f'Propagación de {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/multilayer_analysis/error_propagation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Comparación de voltajes por capa
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mostrar solo las primeras 4 capas
        for i in range(min(4, len(results['float64']['layers']))):
            ax = axes[i//2, i%2]
            
            # Voltaje de referencia (float64)
            ref_layer = results['float64']['layers'][i]
            time = ref_layer['time']
            ref_voltage = np.mean(ref_layer['voltage'], axis=0)
            
            ax.plot(time, ref_voltage, 'k-', label='float64 (ref)', linewidth=2)
            
            # Voltajes de otras precisiones
            for precision in ['float32', 'float16', 'posit16']:
                if precision in results:
                    test_layer = results[precision]['layers'][i]
                    test_voltage = np.mean(test_layer['voltage'], axis=0)
                    ax.plot(time, test_voltage, '--', label=precision, alpha=0.7)
            
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Voltaje promedio (mV)')
            ax.set_title(f'Capa {i} - Voltajes Promedio')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/multilayer_analysis/voltage_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Heatmap de correlaciones
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, precision in enumerate(['float32', 'float16', 'posit16']):
            if precision not in analysis:
                continue
            
            ax = axes[i]
            
            # Crear matriz de correlaciones
            n_layers = len(analysis[precision])
            correlation_matrix = np.zeros((n_layers, n_layers))
            
            for j in range(n_layers):
                for k in range(n_layers):
                    if j == k:
                        correlation_matrix[j, k] = 1.0
                    else:
                        # Usar correlación de la capa actual
                        layer_key = f'layer_{j}'
                        if layer_key in analysis[precision]:
                            correlation_matrix[j, k] = analysis[precision][layer_key]['correlation']
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=ax, xticklabels=range(n_layers), yticklabels=range(n_layers))
            ax.set_title(f'Correlaciones - {precision}')
            ax.set_xlabel('Capa')
            ax.set_ylabel('Capa')
        
        plt.tight_layout()
        plt.savefig('results/multilayer_analysis/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_error_amplification_report(self, analysis):
        """
        Crea reporte de amplificación de errores
        
        Args:
            analysis: Análisis de propagación de errores
        """
        logging.info("=== CREANDO REPORTE DE AMPLIFICACIÓN DE ERRORES ===")
        
        # Crear DataFrame con todos los datos
        report_data = []
        
        for precision in analysis.keys():
            for layer_key in analysis[precision].keys():
                layer_num = int(layer_key.split('_')[1])
                metrics = analysis[precision][layer_key]
                
                row = {
                    'precision': precision,
                    'layer': layer_num,
                    'rmse_voltage': metrics['rmse_voltage'],
                    'max_error': metrics['max_error'],
                    'spike_diff': metrics['spike_diff'],
                    'rate_diff': metrics['rate_diff'],
                    'correlation': metrics['correlation'],
                    'ref_n_spikes': metrics['ref_n_spikes'],
                    'test_n_spikes': metrics['test_n_spikes'],
                    'ref_rate': metrics['ref_rate'],
                    'test_rate': metrics['test_rate']
                }
                
                report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Guardar reporte
        df.to_csv('results/multilayer_analysis/error_amplification_report.csv', index=False)
        
        # Análisis de amplificación
        amplification_analysis = {}
        
        for precision in analysis.keys():
            precision_data = df[df['precision'] == precision]
            
            if len(precision_data) > 1:
                # Calcular tasa de amplificación
                layers = sorted(precision_data['layer'].unique())
                
                rmse_amplification = []
                spike_amplification = []
                
                for i in range(1, len(layers)):
                    prev_layer = precision_data[precision_data['layer'] == layers[i-1]]
                    curr_layer = precision_data[precision_data['layer'] == layers[i]]
                    
                    if len(prev_layer) > 0 and len(curr_layer) > 0:
                        rmse_ratio = curr_layer['rmse_voltage'].iloc[0] / prev_layer['rmse_voltage'].iloc[0]
                        spike_ratio = abs(curr_layer['spike_diff'].iloc[0]) / (abs(prev_layer['spike_diff'].iloc[0]) + 1e-6)
                        
                        rmse_amplification.append(rmse_ratio)
                        spike_amplification.append(spike_ratio)
                
                amplification_analysis[precision] = {
                    'rmse_amplification_mean': np.mean(rmse_amplification) if rmse_amplification else np.nan,
                    'spike_amplification_mean': np.mean(spike_amplification) if spike_amplification else np.nan,
                    'rmse_amplification_std': np.std(rmse_amplification) if rmse_amplification else np.nan,
                    'spike_amplification_std': np.std(spike_amplification) if spike_amplification else np.nan
                }
        
        # Guardar análisis de amplificación
        amplification_df = pd.DataFrame(amplification_analysis).T
        amplification_df.to_csv('results/multilayer_analysis/amplification_analysis.csv')
        
        # Imprimir resumen
        print("\n=== RESUMEN DE AMPLIFICACIÓN DE ERRORES ===")
        for precision, metrics in amplification_analysis.items():
            print(f"\n{precision}:")
            print(f"  Amplificación RMSE: {metrics['rmse_amplification_mean']:.3f} ± {metrics['rmse_amplification_std']:.3f}")
            print(f"  Amplificación Spikes: {metrics['spike_amplification_mean']:.3f} ± {metrics['spike_amplification_std']:.3f}")
        
        return df, amplification_analysis
    
    def run_comprehensive_multilayer_analysis(self):
        """
        Ejecuta análisis comprehensivo de redes multicapa
        """
        logging.info("=== INICIANDO ANÁLISIS COMPREHENSIVO DE REDES MULTICAPA ===")
        
        # Configuraciones de red a probar
        network_configs = [
            {'name': 'small', 'layers': [10, 8, 5]},
            {'name': 'medium', 'layers': [20, 15, 10, 5]},
            {'name': 'deep', 'layers': [30, 25, 20, 15, 10, 5]}
        ]
        
        # Corrientes de entrada
        input_currents = [10.0, 15.0, 20.0] * b2.mV
        
        all_results = {}
        
        for config in network_configs:
            logging.info(f"Probando configuración: {config['name']}")
            
            for I_ext in input_currents:
                logging.info(f"  Con corriente: {I_ext}")
                
                # Ejecutar simulación
                results = self.run_multilayer_simulation(config['layers'], I_ext)
                
                # Analizar propagación de errores
                analysis = self.analyze_error_propagation(results)
                
                # Crear visualizaciones
                self.create_multilayer_visualizations(results, analysis)
                
                # Crear reporte
                df, amplification = self.create_error_amplification_report(analysis)
                
                # Almacenar resultados
                key = f"{config['name']}_{float(I_ext/b2.mV)}"
                all_results[key] = {
                    'config': config,
                    'I_ext': I_ext,
                    'results': results,
                    'analysis': analysis,
                    'report': df,
                    'amplification': amplification
                }
        
        logging.info("=== ANÁLISIS MULTICAPA COMPLETADO ===")
        
        return all_results

def main():
    """Función principal"""
    print("=== ANÁLISIS DE REDES MULTICAPA ===")
    print("Estudiando propagación de errores de precisión en redes feedforward")
    print("1. Redes multicapa con diferentes configuraciones")
    print("2. Análisis de propagación de errores capa a capa")
    print("3. Visualizaciones de raster plots y voltajes")
    print("4. Reporte de amplificación de errores")
    print()
    
    # Crear instancia del analizador
    analyzer = MultilayerNetworkAnalysis(dt=0.1*b2.ms, simulation_time=1000*b2.ms)
    
    # Ejecutar análisis comprehensivo
    results = analyzer.run_comprehensive_multilayer_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- multilayer_network_analysis.log: Log detallado")
    print("- results/multilayer_analysis/raster_plots.png: Raster plots por capa")
    print("- results/multilayer_analysis/error_propagation.png: Propagación de errores")
    print("- results/multilayer_analysis/voltage_comparison.png: Comparación de voltajes")
    print("- results/multilayer_analysis/error_amplification_report.csv: Reporte detallado")
    print("- results/multilayer_analysis/amplification_analysis.csv: Análisis de amplificación")

if __name__ == "__main__":
    main() 