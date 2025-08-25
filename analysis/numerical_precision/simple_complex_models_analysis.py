import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Función de conversión para posit16 (simulada con float32)
def convert16(value):
    """Convierte un valor a precisión de 16 bits (simulado con float32)"""
    return np.float32(value)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_complex_models.log'),
        logging.StreamHandler()
    ]
)

class SimpleComplexModelsAnalysis:
    def __init__(self):
        """Inicializa el análisis de modelos complejos"""
        self.results = {}
        os.makedirs('results/simple_complex_models', exist_ok=True)
        logging.info("SimpleComplexModelsAnalysis inicializado")
    
    def analyze_izhikevich_models(self):
        """Analiza modelos de neurona Izhikevich"""
        logging.info("=== ANALIZANDO MODELOS IZHIKEWICH ===")
        
        # Parámetros para diferentes tipos de neuronas Izhikevich
        neuron_types = {
            'RS': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 2},  # Regular spiking
            'IB': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},  # Intrinsically bursting
            'CH': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},  # Chattering
            'FS': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2}    # Fast spiking
        }
        
        precision_types = ['float64', 'float32', 'float16', 'posit16']
        results = {}
        
        for neuron_type, params in neuron_types.items():
            logging.info(f"Probando neurona {neuron_type}")
            
            neuron_results = {}
            for precision in precision_types:
                try:
                    # Limpiar estado
                    b2.device.reinit()
                    b2.device.activate()
                    
                    # Ecuaciones Izhikevich
                    eqs = '''
                    dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
                    du/dt = a*(b*v - u) / ms : 1
                    I : 1
                    '''
                    
                    # Crear grupo neuronal
                    if precision == 'float64':
                        dtype = np.float64
                    elif precision == 'float32':
                        dtype = np.float32
                    elif precision == 'float16':
                        dtype = np.float16
                    elif precision == 'posit16':
                        dtype = np.float32
                    
                    G = b2.NeuronGroup(1, eqs, threshold='v >= 30', reset='v = c; u += d',
                                      method='euler', dtype=dtype)
                    
                    # Condiciones iniciales
                    G.v = params['c']
                    G.u = params['b'] * params['c']
                    
                    # Aplicar conversión de precisión si es necesario
                    if precision == 'posit16':
                        G.v = convert16(float(G.v[0]))
                        G.u = convert16(float(G.u[0]))
                    
                    # Monitores
                    mon = b2.StateMonitor(G, ['v', 'u'], record=True)
                    spk = b2.SpikeMonitor(G)
                    
                    # Network
                    net = b2.Network(G, mon, spk)
                    
                    # Ejecutar simulación
                    simulation_time = 1000 * b2.ms
                    I_ext = 10.0  # Corriente externa
                    
                    namespace = {
                        'a': params['a'], 'b': params['b'], 'c': params['c'], 'd': params['d'],
                        'I': I_ext
                    }
                    
                    net.run(simulation_time, namespace=namespace)
                    
                    # Extraer resultados
                    voltage = mon.v[0]
                    recovery = mon.u[0]
                    time = mon.t / b2.ms
                    spikes = spk.t / b2.ms
                    
                    neuron_results[precision] = {
                        'voltage': voltage,
                        'recovery': recovery,
                        'time': time,
                        'spikes': spikes,
                        'n_spikes': len(spikes),
                        'firing_rate': len(spikes) / (float(simulation_time / b2.second))
                    }
                    
                    logging.info(f"  {precision}: {len(spikes)} spikes, {neuron_results[precision]['firing_rate']:.2f} Hz")
                    
                except Exception as e:
                    logging.error(f"Error en {neuron_type} con {precision}: {e}")
                    neuron_results[precision] = {
                        'voltage': np.array([]),
                        'recovery': np.array([]),
                        'time': np.array([]),
                        'spikes': np.array([]),
                        'n_spikes': 0,
                        'firing_rate': 0.0,
                        'error': str(e)
                    }
            
            results[neuron_type] = neuron_results
        
        # Crear visualizaciones
        self.plot_izhikevich_results(results)
        
        self.results['izhikevich'] = results
        logging.info("Análisis Izhikevich completado")
    
    def plot_izhikevich_results(self, results):
        """Crea visualizaciones para modelos Izhikevich"""
        logging.info("Creando visualizaciones Izhikevich")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {'float64': 'blue', 'float32': 'green', 'float16': 'orange', 'posit16': 'red'}
        
        for i, neuron_type in enumerate(['RS', 'IB', 'CH', 'FS']):
            ax = axes[i]
            
            for precision, color in colors.items():
                if precision in results[neuron_type]:
                    data = results[neuron_type][precision]
                    if len(data['time']) > 0 and 'error' not in data:
                        ax.plot(data['time'], data['voltage'], color=color, 
                               label=f'{precision}', linewidth=1, alpha=0.8)
            
            ax.set_title(f'Neurona {neuron_type}')
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Potencial (mV)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/simple_complex_models/izhikevich_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear gráfico de tasas de disparo
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(['RS', 'IB', 'CH', 'FS']))
        width = 0.2
        
        for j, precision in enumerate(['float64', 'float32', 'float16', 'posit16']):
            rates = []
            for neuron_type in ['RS', 'IB', 'CH', 'FS']:
                if precision in results[neuron_type]:
                    data = results[neuron_type][precision]
                    if 'error' not in data:
                        rates.append(data['firing_rate'])
                    else:
                        rates.append(0)
                else:
                    rates.append(0)
            
            ax.bar(x_pos + j*width, rates, width, label=precision, alpha=0.8)
        
        ax.set_xlabel('Tipo de Neurona')
        ax.set_ylabel('Tasa de Disparo (Hz)')
        ax.set_title('Comparación de Tasas de Disparo por Precisión')
        ax.set_xticks(x_pos + width*1.5)
        ax.set_xticklabels(['RS', 'IB', 'CH', 'FS'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/simple_complex_models/firing_rates_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones Izhikevich creadas")
    
    def create_summary_report(self):
        """Crea un reporte resumido"""
        logging.info("Creando reporte resumido")
        
        report = []
        report.append("# Análisis de Modelos Neuronales Complejos")
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'izhikevich' in self.results:
            report.append("## Modelos Izhikevich")
            report.append("")
            
            for neuron_type in ['RS', 'IB', 'CH', 'FS']:
                report.append(f"### Neurona {neuron_type}")
                report.append("")
                
                for precision in ['float64', 'float32', 'float16', 'posit16']:
                    if precision in self.results['izhikevich'][neuron_type]:
                        data = self.results['izhikevich'][neuron_type][precision]
                        if 'error' in data:
                            report.append(f"- **{precision}**: Error - {data['error']}")
                        else:
                            report.append(f"- **{precision}**: {data['n_spikes']} spikes, {data['firing_rate']:.2f} Hz")
                report.append("")
        
        # Guardar reporte
        with open('results/simple_complex_models/summary_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        logging.info("Reporte resumido creado")
    
    def run_analysis(self):
        """Ejecuta el análisis completo"""
        logging.info("=== INICIANDO ANÁLISIS DE MODELOS COMPLEJOS ===")
        
        try:
            self.analyze_izhikevich_models()
            self.create_summary_report()
            
            logging.info("=== ANÁLISIS COMPLETADO ===")
            logging.info("Revisa los archivos generados:")
            logging.info("- simple_complex_models.log: Log detallado")
            logging.info("- results/simple_complex_models/summary_report.md: Reporte resumido")
            logging.info("- results/simple_complex_models/izhikevich_comparison.png: Comparación de voltajes")
            logging.info("- results/simple_complex_models/firing_rates_comparison.png: Comparación de tasas")
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")

def main():
    """Función principal"""
    analyzer = SimpleComplexModelsAnalysis()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 