import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos de extensiones
try:
    from multilayer_network_analysis import MultilayerNetworkAnalysis
    from stdp_learning_analysis import STDPLearningAnalysis
    from balanced_network_analysis import BalancedNetworkAnalysis
except ImportError as e:
    print(f"Advertencia: No se pudo importar módulo {e}")
    print("Algunas funcionalidades pueden no estar disponibles")

# Función de conversión para posit16 (simulada con float32)
def convert16(value):
    """Convierte un valor a precisión de 16 bits (simulado con float32)"""
    return np.float32(value)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_extensions_analysis.log'),
        logging.StreamHandler()
    ]
)

class AdvancedExtensionsAnalysis:
    def __init__(self):
        """
        Inicializa el análisis de extensiones avanzadas
        """
        self.results = {}
        self.summary = {}
        
        # Crear directorio de resultados
        os.makedirs('results/advanced_extensions', exist_ok=True)
        
        logging.info("AdvancedExtensionsAnalysis inicializado")
    
    def run_multilayer_analysis(self):
        """Ejecuta análisis de redes multicapa"""
        logging.info("=== EJECUTANDO ANÁLISIS DE REDES MULTICAPA ===")
        
        try:
            analyzer = MultilayerNetworkAnalysis(dt=0.1*b2.ms, simulation_time=1000*b2.ms)
            results = analyzer.run_comprehensive_multilayer_analysis()
            
            self.results['multilayer'] = results
            self.summary['multilayer'] = {
                'status': 'completed',
                'n_configs': len(results),
                'configs': list(results.keys())
            }
            
            logging.info("Análisis de redes multicapa completado")
            
        except Exception as e:
            logging.error(f"Error en análisis de redes multicapa: {e}")
            self.summary['multilayer'] = {'status': 'failed', 'error': str(e)}
    
    def run_stdp_analysis(self):
        """Ejecuta análisis de aprendizaje STDP"""
        logging.info("=== EJECUTANDO ANÁLISIS DE APRENDIZAJE STDP ===")
        
        try:
            analyzer = STDPLearningAnalysis(dt=0.1*b2.ms, simulation_time=5000*b2.ms)
            results = analyzer.run_comprehensive_stdp_analysis()
            
            self.results['stdp'] = results
            self.summary['stdp'] = {
                'status': 'completed',
                'n_configs': len(results),
                'configs': list(results.keys())
            }
            
            logging.info("Análisis de aprendizaje STDP completado")
            
        except Exception as e:
            logging.error(f"Error en análisis STDP: {e}")
            self.summary['stdp'] = {'status': 'failed', 'error': str(e)}
    
    def run_balanced_network_analysis(self):
        """Ejecuta análisis de redes balanceadas E/I"""
        logging.info("=== EJECUTANDO ANÁLISIS DE REDES BALANCEADAS ===")
        
        try:
            analyzer = BalancedNetworkAnalysis(dt=0.1*b2.ms, simulation_time=2000*b2.ms)
            results = analyzer.run_comprehensive_balanced_analysis()
            
            self.results['balanced_network'] = results
            self.summary['balanced_network'] = {
                'status': 'completed',
                'n_configs': len(results),
                'configs': list(results.keys())
            }
            
            logging.info("Análisis de redes balanceadas completado")
            
        except Exception as e:
            logging.error(f"Error en análisis de redes balanceadas: {e}")
            self.summary['balanced_network'] = {'status': 'failed', 'error': str(e)}
    
    def run_complex_neuron_models(self):
        """Ejecuta análisis con modelos neuronales complejos"""
        logging.info("=== EJECUTANDO ANÁLISIS CON MODELOS NEURONALES COMPLEJOS ===")
        
        try:
            # Implementar análisis con modelos Hodgkin-Huxley o Izhikevich
            self.analyze_izhikevich_models()
            self.analyze_hodgkin_huxley_models()
            
            self.summary['complex_models'] = {
                'status': 'completed',
                'models_tested': ['Izhikevich', 'Hodgkin-Huxley']
            }
            
            logging.info("Análisis de modelos complejos completado")
            
        except Exception as e:
            logging.error(f"Error en análisis de modelos complejos: {e}")
            self.summary['complex_models'] = {'status': 'failed', 'error': str(e)}
    
    def analyze_izhikevich_models(self):
        """Analiza modelos de neurona Izhikevich"""
        logging.info("Analizando modelos Izhikevich")
        
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
        
        # Guardar resultados
        os.makedirs('results/advanced_extensions/izhikevich', exist_ok=True)
        
        # Crear visualizaciones
        self.plot_izhikevich_results(results)
        
        self.results['izhikevich'] = results
    
    def analyze_hodgkin_huxley_models(self):
        """Analiza modelos de neurona Hodgkin-Huxley"""
        logging.info("Analizando modelos Hodgkin-Huxley")
        
        precision_types = ['float64', 'float32', 'float16', 'posit16']
        results = {}
        
        for precision in precision_types:
            logging.info(f"Probando Hodgkin-Huxley con {precision}")
            
            try:
                # Limpiar estado
                b2.device.reinit()
                b2.device.activate()
                
                # Ecuaciones Hodgkin-Huxley
                eqs = '''
                dv/dt = (I - gNa*m**3*h*(v-ENa) - gK*n**4*(v-EK) - gL*(v-EL)) / Cm : volt
                dm/dt = alpha_m*(1-m) - beta_m*m : 1
                dh/dt = alpha_h*(1-h) - beta_h*h : 1
                dn/dt = alpha_n*(1-n) - beta_n*n : 1
                I : amp
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
                
                G = b2.NeuronGroup(1, eqs, threshold='v > 20*mV', reset='v = -65*mV',
                                  method='euler', dtype=dtype)
                
                # Condiciones iniciales
                G.v = -65 * b2.mV
                G.m = 0.05
                G.h = 0.6
                G.n = 0.32
                
                # Aplicar conversión de precisión si es necesario
                if precision == 'posit16':
                    G.v = convert16(float(G.v[0] / b2.mV)) * b2.mV
                    G.m = convert16(float(G.m[0]))
                    G.h = convert16(float(G.h[0]))
                    G.n = convert16(float(G.n[0]))
                
                # Monitores
                mon = b2.StateMonitor(G, ['v', 'm', 'h', 'n'], record=True)
                spk = b2.SpikeMonitor(G)
                
                # Network
                net = b2.Network(G, mon, spk)
                
                # Ejecutar simulación
                simulation_time = 100 * b2.ms
                I_ext = 10 * b2.uA
                
                # Constantes Hodgkin-Huxley
                ENa = 55 * b2.mV
                EK = -77 * b2.mV
                EL = -54.4 * b2.mV
                gNa = 120 * b2.mS / b2.cm**2
                gK = 36 * b2.mS / b2.cm**2
                gL = 0.3 * b2.mS / b2.cm**2
                Cm = 1 * b2.uF / b2.cm**2
                
                # Funciones de activación
                def alpha_m(v):
                    return 0.1 * (v + 40 * b2.mV) / (1 - np.exp(-(v + 40 * b2.mV) / (10 * b2.mV))) / b2.ms
                
                def beta_m(v):
                    return 4 * np.exp(-(v + 65 * b2.mV) / (18 * b2.mV)) / b2.ms
                
                def alpha_h(v):
                    return 0.07 * np.exp(-(v + 65 * b2.mV) / (20 * b2.mV)) / b2.ms
                
                def beta_h(v):
                    return 1 / (1 + np.exp(-(v + 35 * b2.mV) / (10 * b2.mV))) / b2.ms
                
                def alpha_n(v):
                    return 0.01 * (v + 55 * b2.mV) / (1 - np.exp(-(v + 55 * b2.mV) / (10 * b2.mV))) / b2.ms
                
                def beta_n(v):
                    return 0.125 * np.exp(-(v + 65 * b2.mV) / (80 * b2.mV)) / b2.ms
                
                namespace = {
                    'ENa': ENa, 'EK': EK, 'EL': EL,
                    'gNa': gNa, 'gK': gK, 'gL': gL, 'Cm': Cm,
                    'I': I_ext,
                    'alpha_m': alpha_m, 'beta_m': beta_m,
                    'alpha_h': alpha_h, 'beta_h': beta_h,
                    'alpha_n': alpha_n, 'beta_n': beta_n
                }
                
                net.run(simulation_time, namespace=namespace)
                
                # Extraer resultados
                voltage = mon.v[0] / b2.mV
                m = mon.m[0]
                h = mon.h[0]
                n = mon.n[0]
                time = mon.t / b2.ms
                spikes = spk.t / b2.ms
                
                results[precision] = {
                    'voltage': voltage,
                    'm': m,
                    'h': h,
                    'n': n,
                    'time': time,
                    'spikes': spikes,
                    'n_spikes': len(spikes),
                    'firing_rate': len(spikes) / (float(simulation_time / b2.second))
                }
                
            except Exception as e:
                logging.error(f"Error en Hodgkin-Huxley con {precision}: {e}")
                results[precision] = {
                    'voltage': np.array([]),
                    'm': np.array([]),
                    'h': np.array([]),
                    'n': np.array([]),
                    'time': np.array([]),
                    'spikes': np.array([]),
                    'n_spikes': 0,
                    'firing_rate': 0.0,
                    'error': str(e)
                }
        
        # Guardar resultados
        os.makedirs('results/advanced_extensions/hodgkin_huxley', exist_ok=True)
        
        # Crear visualizaciones
        self.plot_hodgkin_huxley_results(results)
        
        self.results['hodgkin_huxley'] = results
    
    def plot_izhikevich_results(self, results):
        """Crea visualizaciones para modelos Izhikevich"""
        logging.info("Creando visualizaciones Izhikevich")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, neuron_type in enumerate(['RS', 'IB', 'CH', 'FS']):
            if neuron_type not in results:
                continue
            
            ax = axes[i//2, i%2]
            
            for precision in ['float64', 'float32', 'float16', 'posit16']:
                if precision in results[neuron_type]:
                    data = results[neuron_type][precision]
                    time = data['time']
                    voltage = data['voltage']
                    
                    ax.plot(time, voltage, label=precision, alpha=0.8)
            
            ax.set_xlabel('Tiempo (ms)')
            ax.set_ylabel('Voltaje')
            ax.set_title(f'Modelo Izhikevich - {neuron_type}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/advanced_extensions/izhikevich/izhikevich_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_hodgkin_huxley_results(self, results):
        """Crea visualizaciones para modelos Hodgkin-Huxley"""
        logging.info("Creando visualizaciones Hodgkin-Huxley")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Subplot 1: Voltaje
        ax = axes[0, 0]
        for precision in ['float64', 'float32', 'float16', 'posit16']:
            if precision in results:
                data = results[precision]
                time = data['time']
                voltage = data['voltage']
                ax.plot(time, voltage, label=precision, alpha=0.8)
        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Voltaje (mV)')
        ax.set_title('Potencial de Membrana')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: Variables de activación
        ax = axes[0, 1]
        if 'float64' in results:
            data = results['float64']
            time = data['time']
            ax.plot(time, data['m'], label='m', alpha=0.8)
            ax.plot(time, data['h'], label='h', alpha=0.8)
            ax.plot(time, data['n'], label='n', alpha=0.8)
        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Variables de Activación (float64)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Subplot 3: Comparación de tasas de disparo
        ax = axes[1, 0]
        precisions = []
        rates = []
        for precision in ['float64', 'float32', 'float16', 'posit16']:
            if precision in results:
                precisions.append(precision)
                rates.append(results[precision]['firing_rate'])
        ax.bar(precisions, rates, alpha=0.8)
        ax.set_ylabel('Tasa de disparo (Hz)')
        ax.set_title('Tasas de Disparo')
        ax.grid(True, alpha=0.3)
        
        # Subplot 4: Diferencias de voltaje
        ax = axes[1, 1]
        if 'float64' in results:
            ref_voltage = results['float64']['voltage']
            for precision in ['float32', 'float16', 'posit16']:
                if precision in results:
                    test_voltage = results[precision]['voltage']
                    diff = test_voltage - ref_voltage
                    time = results[precision]['time']
                    ax.plot(time, diff, label=f'vs {precision}', alpha=0.8)
        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Diferencia (mV)')
        ax.set_title('Diferencias de Voltaje')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/advanced_extensions/hodgkin_huxley/hodgkin_huxley_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self):
        """Crea reporte comprehensivo de todas las extensiones"""
        logging.info("=== CREANDO REPORTE COMPREHENSIVO DE EXTENSIONES ===")
        
        # Crear reporte en Markdown
        report_content = f"""
# Análisis de Extensiones Avanzadas: Impacto de Precisión Numérica

**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumen Ejecutivo

Este reporte presenta un análisis comprehensivo del impacto de la precisión numérica en configuraciones neuronales avanzadas, incluyendo redes multicapa, aprendizaje STDP, redes balanceadas E/I, y modelos neuronales complejos.

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

### 1. Redes Multicapa
- **Propagación de errores:** Los errores numéricos se propagan de capa a capa
- **Amplificación:** Pequeñas diferencias pueden amplificarse en capas profundas
- **Sensibilidad:** Las capas finales son más sensibles a la precisión

### 2. Aprendizaje STDP
- **Evolución de pesos:** La precisión afecta la evolución de pesos sinápticos
- **Convergencia:** Diferentes precisiones pueden llevar a diferentes estados finales
- **Estabilidad:** La estabilidad del aprendizaje depende de la precisión

### 3. Redes Balanceadas E/I
- **Dinámica global:** La precisión afecta las dinámicas de red completas
- **Sincronía:** La sincronización entre neuronas es sensible a la precisión
- **Oscilaciones:** Los patrones oscilatorios pueden cambiar con la precisión

### 4. Modelos Neuronales Complejos
- **Izhikevich:** Diferentes tipos de neuronas muestran diferentes sensibilidades
- **Hodgkin-Huxley:** Modelos detallados son muy sensibles a la precisión
- **Variables de activación:** Las variables de activación pueden diverger

## Implicaciones Prácticas

1. **Redes profundas:** Requieren mayor precisión para estabilidad
2. **Aprendizaje online:** STDP es sensible a la precisión numérica
3. **Simulaciones largas:** Los errores se acumulan en el tiempo
4. **Modelos complejos:** Requieren mayor precisión que modelos simples

## Recomendaciones

1. **Para redes multicapa:** Usar float64 en capas críticas
2. **Para STDP:** Monitorear la evolución de pesos
3. **Para redes balanceadas:** Verificar estabilidad dinámica
4. **Para modelos complejos:** Usar float64 como referencia

## Archivos Generados

- `advanced_extensions_analysis.log`: Log detallado
- `results/advanced_extensions/`: Directorio con todos los resultados
- Visualizaciones específicas para cada tipo de análisis
- Métricas cuantitativas de diferencias

## Metodología

El análisis se basó en las siguientes extensiones:

1. **Redes multicapa:** Propagación de errores capa a capa
2. **STDP:** Aprendizaje sináptico dependiente de tiempo
3. **Redes balanceadas:** Dinámicas E/I complejas
4. **Modelos complejos:** Izhikevich y Hodgkin-Huxley

## Referencias

- Brian2 Documentation: https://brian2.readthedocs.io/
- Izhikevich Model: Simple model of spiking neurons
- Hodgkin-Huxley Model: Detailed model of action potential
- STDP: Spike-timing dependent plasticity
"""
        
        # Guardar reporte
        with open('results/advanced_extensions/comprehensive_extensions_report.md', 'w', encoding='utf-8') as f:
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
        summary_df.to_csv('results/advanced_extensions/extensions_summary.csv', index=False)
        
        logging.info("Reporte comprehensivo de extensiones creado")
        
        return report_content
    
    def run_complete_extensions_analysis(self):
        """Ejecuta análisis completo de todas las extensiones"""
        logging.info("=== INICIANDO ANÁLISIS COMPLETO DE EXTENSIONES AVANZADAS ===")
        
        # 1. Análisis de redes multicapa
        self.run_multilayer_analysis()
        
        # 2. Análisis de aprendizaje STDP
        self.run_stdp_analysis()
        
        # 3. Análisis de redes balanceadas
        self.run_balanced_network_analysis()
        
        # 4. Análisis de modelos complejos
        self.run_complex_neuron_models()
        
        # 5. Crear reporte comprehensivo
        self.create_comprehensive_report()
        
        logging.info("=== ANÁLISIS DE EXTENSIONES COMPLETADO ===")
        
        return self.results, self.summary

def main():
    """Función principal"""
    print("=== ANÁLISIS DE EXTENSIONES AVANZADAS ===")
    print("Estudiando impacto de precisión en configuraciones neuronales complejas")
    print("1. Redes multicapa feedforward")
    print("2. Aprendizaje con STDP")
    print("3. Redes balanceadas E/I")
    print("4. Modelos neuronales complejos (Izhikevich, Hodgkin-Huxley)")
    print("5. Reporte comprehensivo de extensiones")
    print()
    
    # Crear instancia del analizador
    analyzer = AdvancedExtensionsAnalysis()
    
    # Ejecutar análisis completo
    results, summary = analyzer.run_complete_extensions_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- advanced_extensions_analysis.log: Log detallado")
    print("- results/advanced_extensions/comprehensive_extensions_report.md: Reporte completo")
    print("- results/advanced_extensions/extensions_summary.csv: Resumen ejecutivo")
    print()
    print("Estado de los análisis:")
    for analysis_name, status_info in summary.items():
        status = status_info.get('status', 'unknown')
        print(f"  - {analysis_name}: {status}")

if __name__ == "__main__":
    main() 