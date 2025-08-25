import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from posit_wrapper import convert16
import pandas as pd

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brian2_debug_callbacks.log'),
        logging.StreamHandler()
    ]
)

# Configurar Brian2 para logging detallado
b2.prefs.logging.file_log_level = 'DIAGNOSTIC'
b2.prefs.logging.delete_log_on_exit = False
b2.prefs.codegen.target = 'numpy'

class Brian2DebugCallbacks:
    def __init__(self, dt=0.1*b2.ms, simulation_time=100*b2.ms):
        self.dt = dt
        self.simulation_time = simulation_time
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        self.I_ext = 20.0 * b2.mV
        
        # Almacenamiento de datos de debug
        self.debug_data = {
            'float64': [],
            'float16': [],
            'posit16': []
        }
        
        # Contadores de pasos
        self.step_counters = {
            'float64': 0,
            'float16': 0,
            'posit16': 0
        }
        
        logging.info(f"Brian2DebugCallbacks inicializado")
    
    def create_debug_callback(self, precision_type):
        """Crea un callback de debug para un tipo de precisión específico"""
        def debug_callback():
            # Obtener el grupo neuronal actual
            if precision_type == 'float64':
                group = self.G64
            elif precision_type == 'float16':
                group = self.G16
            elif precision_type == 'posit16':
                group = self.G_posit
            
            # Extraer información del estado actual
            v_current = float(group.v[0] / b2.mV)
            t_current = float(b2.defaultclock.t / b2.ms)
            step = self.step_counters[precision_type]
            
            # Calcular derivada numérica (aproximación)
            if step > 0:
                v_prev = self.debug_data[precision_type][-1][2]
                dt_ms = float(self.dt / b2.ms)
                dv_dt = (v_current - v_prev) / dt_ms
            else:
                dv_dt = 0.0
            
            # Almacenar datos de debug
            debug_info = {
                'step': step,
                'time_ms': t_current,
                'voltage_mV': v_current,
                'dv_dt_mV_ms': dv_dt,
                'tau_ms': float(self.tau / b2.ms),
                'I_ext_mV': float(self.I_ext / b2.mV),
                'V_rest_mV': float(self.V_rest / b2.mV)
            }
            
            self.debug_data[precision_type].append(debug_info)
            self.step_counters[precision_type] += 1
            
            # Log cada 50 pasos para no saturar
            if step % 50 == 0:
                logging.debug(f"{precision_type} - Paso {step}: t={t_current:.3f}ms, v={v_current:.6f}mV, dv/dt={dv_dt:.6f}mV/ms")
        
        return debug_callback
    
    def create_spike_detection_callback(self, precision_type):
        """Crea un callback para detectar spikes y analizar el comportamiento"""
        def spike_detection_callback():
            if precision_type == 'float64':
                group = self.G64
            elif precision_type == 'float16':
                group = self.G16
            elif precision_type == 'posit16':
                group = self.G_posit
            
            v_current = float(group.v[0] / b2.mV)
            t_current = float(b2.defaultclock.t / b2.ms)
            Vth_mV = float(self.Vth / b2.mV)
            
            # Detectar si está cerca del umbral
            threshold_proximity = abs(v_current - Vth_mV)
            
            if threshold_proximity < 1.0:  # Dentro de 1mV del umbral
                logging.info(f"{precision_type} - CERCA DEL UMBRAL: t={t_current:.3f}ms, v={v_current:.6f}mV, proximidad={threshold_proximity:.6f}mV")
            
            # Detectar valores extremos o anómalos
            if v_current > -30 or v_current < -90:
                logging.warning(f"{precision_type} - VALOR EXTREMO: t={t_current:.3f}ms, v={v_current:.6f}mV")
        
        return spike_detection_callback
    
    def create_divergence_detection_callback(self):
        """Crea un callback para detectar divergencias entre precisiones"""
        def divergence_callback():
            # Solo ejecutar si tenemos datos de al menos dos precisiones
            if len(self.debug_data['float64']) > 0 and len(self.debug_data['float16']) > 0:
                # Comparar el último paso
                v64 = self.debug_data['float64'][-1]['voltage_mV']
                v16 = self.debug_data['float16'][-1]['voltage_mV']
                t_current = self.debug_data['float64'][-1]['time_ms']
                
                diff = abs(v64 - v16)
                if diff > 0.1:  # Umbral de divergencia
                    logging.warning(f"DIVERGENCIA DETECTADA: t={t_current:.3f}ms, diff={diff:.6f}mV (float64={v64:.6f}, float16={v16:.6f})")
        
        return divergence_callback
    
    def run_simulation_with_callbacks(self):
        """Ejecuta simulación con múltiples callbacks de debug"""
        logging.info("=== INICIANDO SIMULACIÓN CON CALLBACKS DE DEBUG ===")
        
        eqs = '''
        dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
        '''
        
        # Simulación float64 con callbacks
        logging.info("Ejecutando simulación float64 con callbacks...")
        b2.device.reinit()
        b2.device.activate()
        
        self.G64 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                 refractory=5*b2.ms, method='euler', dtype=np.float64)
        self.G64.v = self.V_rest
        
        # Crear callbacks
        debug_callback_64 = self.create_debug_callback('float64')
        spike_callback_64 = self.create_spike_detection_callback('float64')
        
        # NetworkOperation para callbacks
        debug_op_64 = b2.NetworkOperation(debug_callback_64, dt=self.dt)
        spike_op_64 = b2.NetworkOperation(spike_callback_64, dt=self.dt)
        
        # Monitores
        mon64 = b2.StateMonitor(self.G64, 'v', record=True)
        spk64 = b2.SpikeMonitor(self.G64)
        
        # Network
        net64 = b2.Network(self.G64, debug_op_64, spike_op_64, mon64, spk64)
        
        # Ejecutar simulación
        net64.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        logging.info(f"Simulación float64 completada. {len(self.debug_data['float64'])} pasos registrados")
        
        # Simulación float16 con callbacks
        logging.info("Ejecutando simulación float16 con callbacks...")
        b2.device.reinit()
        b2.device.activate()
        
        self.G16 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                 refractory=5*b2.ms, method='euler', dtype=np.float16)
        self.G16.v = self.V_rest
        
        # Crear callbacks
        debug_callback_16 = self.create_debug_callback('float16')
        spike_callback_16 = self.create_spike_detection_callback('float16')
        
        # NetworkOperation para callbacks
        debug_op_16 = b2.NetworkOperation(debug_callback_16, dt=self.dt)
        spike_op_16 = b2.NetworkOperation(spike_callback_16, dt=self.dt)
        
        # Monitores
        mon16 = b2.StateMonitor(self.G16, 'v', record=True)
        spk16 = b2.SpikeMonitor(self.G16)
        
        # Network
        net16 = b2.Network(self.G16, debug_op_16, spike_op_16, mon16, spk16)
        
        # Ejecutar simulación
        net16.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        logging.info(f"Simulación float16 completada. {len(self.debug_data['float16'])} pasos registrados")
        
        return mon64, spk64, mon16, spk16
    
    def run_simulation_with_divergence_detection(self):
        """Ejecuta simulaciones paralelas con detección de divergencias en tiempo real"""
        logging.info("=== INICIANDO SIMULACIÓN CON DETECCIÓN DE DIVERGENCIAS EN TIEMPO REAL ===")
        
        eqs = '''
        dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
        '''
        
        # Crear grupos neuronales
        b2.device.reinit()
        b2.device.activate()
        
        self.G64 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                 refractory=5*b2.ms, method='euler', dtype=np.float64)
        self.G64.v = self.V_rest
        
        self.G16 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                                 refractory=5*b2.ms, method='euler', dtype=np.float16)
        self.G16.v = self.V_rest
        
        # Callbacks de debug
        debug_callback_64 = self.create_debug_callback('float64')
        debug_callback_16 = self.create_debug_callback('float16')
        divergence_callback = self.create_divergence_detection_callback()
        
        # NetworkOperations
        debug_op_64 = b2.NetworkOperation(debug_callback_64, dt=self.dt)
        debug_op_16 = b2.NetworkOperation(debug_callback_16, dt=self.dt)
        divergence_op = b2.NetworkOperation(divergence_callback, dt=self.dt)
        
        # Monitores
        mon64 = b2.StateMonitor(self.G64, 'v', record=True)
        mon16 = b2.StateMonitor(self.G16, 'v', record=True)
        spk64 = b2.SpikeMonitor(self.G64)
        spk16 = b2.SpikeMonitor(self.G16)
        
        # Network combinado
        net = b2.Network(self.G64, self.G16, debug_op_64, debug_op_16, 
                        divergence_op, mon64, mon16, spk64, spk16)
        
        # Ejecutar simulación
        net.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        logging.info("Simulación con detección de divergencias completada")
        
        return mon64, spk64, mon16, spk16
    
    def analyze_debug_data(self):
        """Analiza los datos de debug recopilados"""
        logging.info("=== ANALIZANDO DATOS DE DEBUG ===")
        
        os.makedirs('results/brian2_debug', exist_ok=True)
        
        # Convertir a DataFrames
        df_float64 = pd.DataFrame(self.debug_data['float64'])
        df_float16 = pd.DataFrame(self.debug_data['float16'])
        
        # Análisis de divergencias
        if len(df_float64) > 0 and len(df_float16) > 0:
            min_len = min(len(df_float64), len(df_float16))
            
            # Calcular diferencias
            df_float64_subset = df_float64.head(min_len)
            df_float16_subset = df_float16.head(min_len)
            
            df_float64_subset['voltage_diff'] = df_float64_subset['voltage_mV'] - df_float16_subset['voltage_mV']
            df_float64_subset['abs_diff'] = abs(df_float64_subset['voltage_diff'])
            
            # Encontrar puntos de divergencia
            divergence_threshold = 0.1
            divergence_points = df_float64_subset[df_float64_subset['abs_diff'] > divergence_threshold]
            
            if len(divergence_points) > 0:
                logging.warning(f"Se encontraron {len(divergence_points)} puntos de divergencia")
                logging.warning(f"Primera divergencia en paso {divergence_points.iloc[0]['step']}")
                logging.warning(f"Tiempo de primera divergencia: {divergence_points.iloc[0]['time_ms']:.3f}ms")
            else:
                logging.info("No se detectaron divergencias significativas")
            
            # Guardar análisis
            df_float64_subset.to_csv('results/brian2_debug/divergence_analysis.csv', index=False)
            
            # Gráficas de análisis
            self.plot_debug_analysis(df_float64_subset, df_float16_subset)
        
        return df_float64, df_float16
    
    def plot_debug_analysis(self, df_float64, df_float16):
        """Genera gráficas de análisis de debug"""
        logging.info("=== GENERANDO GRÁFICAS DE ANÁLISIS DE DEBUG ===")
        
        plt.figure(figsize=(15, 12))
        
        # Subplot 1: Comparación de voltajes
        plt.subplot(3, 2, 1)
        plt.plot(df_float64['time_ms'], df_float64['voltage_mV'], 'k-', label='float64', linewidth=2)
        plt.plot(df_float16['time_ms'], df_float16['voltage_mV'], 'b-', label='float16', linewidth=1, alpha=0.7)
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (mV)')
        plt.title('Comparación de Voltajes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Diferencias
        plt.subplot(3, 2, 2)
        plt.plot(df_float64['time_ms'], df_float64['voltage_diff'], 'r-', label='float64 - float16')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Diferencia (mV)')
        plt.title('Diferencias de Voltaje')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Derivadas
        plt.subplot(3, 2, 3)
        plt.plot(df_float64['time_ms'], df_float64['dv_dt_mV_ms'], 'k-', label='float64', linewidth=2)
        plt.plot(df_float16['time_ms'], df_float16['dv_dt_mV_ms'], 'b-', label='float16', linewidth=1, alpha=0.7)
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('dV/dt (mV/ms)')
        plt.title('Derivadas de Voltaje')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Error absoluto
        plt.subplot(3, 2, 4)
        plt.plot(df_float64['time_ms'], df_float64['abs_diff'], 'g-', label='Error absoluto')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Error absoluto (mV)')
        plt.title('Error Absoluto')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Error acumulado
        plt.subplot(3, 2, 5)
        cum_error = np.cumsum(df_float64['abs_diff'])
        plt.plot(df_float64['time_ms'], cum_error, 'm-', label='Error acumulado')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Error acumulado (mV)')
        plt.title('Error Acumulado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Histograma de diferencias
        plt.subplot(3, 2, 6)
        plt.hist(df_float64['voltage_diff'], bins=50, alpha=0.7, label='Distribución de diferencias')
        plt.xlabel('Diferencia (mV)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Diferencias')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/brian2_debug/debug_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_step_by_step_logging(self):
        """Crea logging paso a paso más detallado"""
        logging.info("=== CREANDO LOGGING PASO A PASO DETALLADO ===")
        
        eqs = '''
        dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
        '''
        
        # Simulación con logging cada paso
        b2.device.reinit()
        b2.device.activate()
        
        G = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                          refractory=5*b2.ms, method='euler', dtype=np.float64)
        G.v = self.V_rest
        
        # Callback para logging detallado
        step_data = []
        def detailed_logging():
            v_current = float(G.v[0] / b2.mV)
            t_current = float(b2.defaultclock.t / b2.ms)
            
            # Calcular valores teóricos
            V_rest_mV = float(self.V_rest / b2.mV)
            I_ext_mV = float(self.I_ext / b2.mV)
            tau_ms = float(self.tau / b2.ms)
            
            # Derivada teórica
            dv_dt_theoretical = (I_ext_mV - (v_current - V_rest_mV)) / tau_ms
            
            step_info = {
                'step': len(step_data),
                'time_ms': t_current,
                'voltage_mV': v_current,
                'dv_dt_theoretical_mV_ms': dv_dt_theoretical,
                'V_rest_mV': V_rest_mV,
                'I_ext_mV': I_ext_mV,
                'tau_ms': tau_ms
            }
            
            step_data.append(step_info)
            
            # Log cada 10 pasos para análisis detallado
            if len(step_data) % 10 == 0:
                logging.info(f"PASO {len(step_data)}: t={t_current:.3f}ms, v={v_current:.6f}mV, dv/dt_teo={dv_dt_theoretical:.6f}mV/ms")
        
        log_op = b2.NetworkOperation(detailed_logging, dt=self.dt)
        mon = b2.StateMonitor(G, 'v', record=True)
        spk = b2.SpikeMonitor(G)
        net = b2.Network(G, log_op, mon, spk)
        
        # Ejecutar simulación
        net.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        # Guardar datos detallados
        df_detailed = pd.DataFrame(step_data)
        df_detailed.to_csv('results/brian2_debug/step_by_step_detailed.csv', index=False)
        
        logging.info(f"Logging detallado completado. {len(step_data)} pasos registrados")
        
        return df_detailed, mon, spk
    
    def run_complete_debug_analysis(self):
        """Ejecuta análisis completo de debug"""
        logging.info("=== INICIANDO ANÁLISIS COMPLETO DE DEBUG ===")
        
        # 1. Simulación con callbacks de debug
        mon64, spk64, mon16, spk16 = self.run_simulation_with_callbacks()
        
        # 2. Simulación con detección de divergencias
        # mon64_div, spk64_div, mon16_div, spk16_div = self.run_simulation_with_divergence_detection()
        
        # 3. Logging paso a paso detallado
        df_detailed, mon_detailed, spk_detailed = self.create_step_by_step_logging()
        
        # 4. Análisis de datos de debug
        df_float64, df_float16 = self.analyze_debug_data()
        
        # 5. Resumen final
        logging.info("=== RESUMEN FINAL DE DEBUG ===")
        logging.info(f"Archivos generados en results/brian2_debug/")
        logging.info(f"Log detallado guardado en brian2_debug_callbacks.log")
        
        return df_float64, df_float16, df_detailed

def main():
    """Función principal"""
    print("=== DEPURACIÓN CON BRIAN2 CALLBACKS ===")
    print("Implementando técnicas avanzadas de depuración con Brian2")
    print("1. Callbacks de debug durante la simulación")
    print("2. Detección de divergencias en tiempo real")
    print("3. Logging paso a paso detallado")
    print("4. Análisis comparativo con callbacks")
    print()
    
    # Crear instancia del debugger
    debugger = Brian2DebugCallbacks(dt=0.1*b2.ms, simulation_time=50*b2.ms)
    
    # Ejecutar análisis completo
    df_float64, df_float16, df_detailed = debugger.run_complete_debug_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- brian2_debug_callbacks.log: Log detallado")
    print("- results/brian2_debug/divergence_analysis.csv: Análisis de divergencias")
    print("- results/brian2_debug/step_by_step_detailed.csv: Logging paso a paso")
    print("- results/brian2_debug/debug_analysis.png: Gráficas de análisis")

if __name__ == "__main__":
    main() 