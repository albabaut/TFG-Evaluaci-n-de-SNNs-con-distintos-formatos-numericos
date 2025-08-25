import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from posit_wrapper import convert16
import pandas as pd
from datetime import datetime

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_numerical.log'),
        logging.StreamHandler()
    ]
)

# Configurar Brian2 para logging detallado
b2.prefs.logging.file_log_level = 'DIAGNOSTIC'
b2.prefs.logging.delete_log_on_exit = False
b2.prefs.codegen.target = 'numpy'

class NumericalDebugger:
    def __init__(self, dt=0.1*b2.ms, simulation_time=100*b2.ms):
        self.dt = dt
        self.simulation_time = simulation_time
        self.n_steps = int(simulation_time / dt)
        
        # Parámetros de la neurona LIF
        self.tau = 10 * b2.ms
        self.V_rest = -70 * b2.mV
        self.Vth = -50 * b2.mV
        self.Vreset = -70 * b2.mV
        self.I_ext = 20.0 * b2.mV
        
        # Almacenamiento de datos
        self.float64_data = []
        self.float16_data = []
        self.posit16_data = []
        self.manual_float64_data = []
        self.manual_float16_data = []
        self.manual_posit16_data = []
        
        logging.info(f"NumericalDebugger inicializado con {self.n_steps} pasos de {dt}")
    
    def run_brian2_simulation_with_logging(self):
        """Ejecuta simulación Brian2 con logging paso a paso"""
        logging.info("=== INICIANDO SIMULACIÓN BRIAN2 CON LOGGING PASO A PASO ===")
        
        eqs = '''
        dv/dt = (-(V_rest - v) + I_ext) / tau : volt (unless refractory)
        '''
        
        # Simulación float64 (referencia)
        logging.info("Ejecutando simulación float64...")
        b2.device.reinit()
        b2.device.activate()
        
        G64 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                            refractory=5*b2.ms, method='euler', dtype=np.float64)
        G64.v = self.V_rest
        
        # Callback para logging cada paso
        step_counter = {'i': 0}
        def log_step():
            v_current = float(G64.v[0] / b2.mV)
            t_current = float(b2.defaultclock.t / b2.ms)
            self.float64_data.append((step_counter['i'], t_current, v_current))
            
            if step_counter['i'] % 100 == 0:  # Log cada 100 pasos para no saturar
                logging.debug(f"Paso {step_counter['i']}: t={t_current:.3f}ms, v={v_current:.6f}mV")
            step_counter['i'] += 1
        
        log_operation = b2.NetworkOperation(log_step, dt=self.dt)
        mon64 = b2.StateMonitor(G64, 'v', record=True)
        spk64 = b2.SpikeMonitor(G64)
        net64 = b2.Network(G64, log_operation, mon64, spk64)
        
        net64.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        logging.info(f"Simulación float64 completada. {len(self.float64_data)} pasos registrados")
        
        # Simulación float16
        logging.info("Ejecutando simulación float16...")
        b2.device.reinit()
        b2.device.activate()
        
        G16 = b2.NeuronGroup(1, eqs, threshold='v > Vth', reset='v = Vreset',
                            refractory=5*b2.ms, method='euler', dtype=np.float16)
        G16.v = self.V_rest
        
        step_counter = {'i': 0}
        def log_step_float16():
            v_current = float(G16.v[0] / b2.mV)
            t_current = float(b2.defaultclock.t / b2.ms)
            self.float16_data.append((step_counter['i'], t_current, v_current))
            
            if step_counter['i'] % 100 == 0:
                logging.debug(f"Paso {step_counter['i']}: t={t_current:.3f}ms, v={v_current:.6f}mV (float16)")
            step_counter['i'] += 1
        
        log_operation = b2.NetworkOperation(log_step_float16, dt=self.dt)
        mon16 = b2.StateMonitor(G16, 'v', record=True)
        spk16 = b2.SpikeMonitor(G16)
        net16 = b2.Network(G16, log_operation, mon16, spk16)
        
        net16.run(self.simulation_time, namespace={
            'I_ext': self.I_ext, 'tau': self.tau, 'V_rest': self.V_rest, 
            'Vth': self.Vth, 'Vreset': self.Vreset
        })
        
        logging.info(f"Simulación float16 completada. {len(self.float16_data)} pasos registrados")
        
        return mon64, spk64, mon16, spk16
    
    def run_manual_numpy_simulation(self):
        """Simulación manual con NumPy para control total"""
        logging.info("=== INICIANDO SIMULACIÓN MANUAL CON NUMPY ===")
        
        dt = float(self.dt / b2.ms)  # Convertir a float
        tau = float(self.tau / b2.ms)
        V_rest = float(self.V_rest / b2.mV)
        Vth = float(self.Vth / b2.mV)
        Vreset = float(self.Vreset / b2.mV)
        I_ext = float(self.I_ext / b2.mV)
        
        # Simulación float64 manual
        logging.info("Ejecutando simulación manual float64...")
        v_float64 = V_rest
        spikes_float64 = []
        
        for step in range(self.n_steps):
            t = step * dt
            
            # Cálculo del cambio de voltaje
            dv = (I_ext - (v_float64 - V_rest)) / tau * dt
            v_float64 += dv
            
            # Detección de spike
            if v_float64 > Vth:
                spikes_float64.append(t)
                v_float64 = Vreset
            
            self.manual_float64_data.append((step, t, v_float64))
            
            if step % 100 == 0:
                logging.debug(f"Manual float64 - Paso {step}: t={t:.3f}ms, v={v_float64:.6f}mV")
        
        # Simulación float16 manual
        logging.info("Ejecutando simulación manual float16...")
        v_float16 = np.float16(V_rest)
        spikes_float16 = []
        
        for step in range(self.n_steps):
            t = step * dt
            
            # Cálculo con conversión a float16 en cada operación
            dv = np.float16((np.float16(I_ext) - np.float16(v_float16 - V_rest)) / np.float16(tau) * np.float16(dt))
            v_float16 = np.float16(v_float16 + dv)
            
            # Detección de spike
            if v_float16 > np.float16(Vth):
                spikes_float16.append(t)
                v_float16 = np.float16(Vreset)
            
            self.manual_float16_data.append((step, t, float(v_float16)))
            
            if step % 100 == 0:
                logging.debug(f"Manual float16 - Paso {step}: t={t:.3f}ms, v={float(v_float16):.6f}mV")
        
        # Simulación posit16 manual
        logging.info("Ejecutando simulación manual posit16...")
        v_posit16 = convert16(V_rest)
        spikes_posit16 = []
        
        for step in range(self.n_steps):
            t = step * dt
            
            # Cálculo con conversión a posit16 en cada operación
            dv = convert16((I_ext - (v_posit16 - V_rest)) / tau * dt)
            v_posit16 = convert16(v_posit16 + dv)
            
            # Detección de spike
            if v_posit16 > convert16(Vth):
                spikes_posit16.append(t)
                v_posit16 = convert16(Vreset)
            
            self.manual_posit16_data.append((step, t, float(v_posit16)))
            
            if step % 100 == 0:
                logging.debug(f"Manual posit16 - Paso {step}: t={t:.3f}ms, v={float(v_posit16):.6f}mV")
        
        logging.info(f"Simulaciones manuales completadas")
        logging.info(f"  - Spikes float64: {len(spikes_float64)}")
        logging.info(f"  - Spikes float16: {len(spikes_float16)}")
        logging.info(f"  - Spikes posit16: {len(spikes_posit16)}")
        
        return spikes_float64, spikes_float16, spikes_posit16
    
    def detect_divergence_points(self, tolerance=0.01):
        """Detecta puntos de divergencia entre simulaciones"""
        logging.info("=== DETECTANDO PUNTOS DE DIVERGENCIA ===")
        
        # Comparar Brian2 float64 vs float16
        if len(self.float64_data) > 0 and len(self.float16_data) > 0:
            min_len = min(len(self.float64_data), len(self.float16_data))
            v64 = np.array([self.float64_data[i][2] for i in range(min_len)])
            v16 = np.array([self.float16_data[i][2] for i in range(min_len)])
            
            diff = np.abs(v64 - v16)
            divergence_idx = np.argmax(diff > tolerance)
            
            if diff[divergence_idx] > tolerance:
                logging.warning(f"Divergencia Brian2 float64 vs float16 detectada en paso {divergence_idx}")
                logging.warning(f"  Tiempo: {self.float64_data[divergence_idx][1]:.3f}ms")
                logging.warning(f"  Diferencia: {diff[divergence_idx]:.6f}mV")
            else:
                logging.info("No se detectaron divergencias significativas entre float64 y float16")
        
        # Comparar simulaciones manuales
        if len(self.manual_float64_data) > 0 and len(self.manual_float16_data) > 0:
            min_len = min(len(self.manual_float64_data), len(self.manual_float16_data))
            v64_manual = np.array([self.manual_float64_data[i][2] for i in range(min_len)])
            v16_manual = np.array([self.manual_float16_data[i][2] for i in range(min_len)])
            
            diff_manual = np.abs(v64_manual - v16_manual)
            divergence_idx_manual = np.argmax(diff_manual > tolerance)
            
            if diff_manual[divergence_idx_manual] > tolerance:
                logging.warning(f"Divergencia manual float64 vs float16 detectada en paso {divergence_idx_manual}")
                logging.warning(f"  Tiempo: {self.manual_float64_data[divergence_idx_manual][1]:.3f}ms")
                logging.warning(f"  Diferencia: {diff_manual[divergence_idx_manual]:.6f}mV")
            else:
                logging.info("No se detectaron divergencias significativas en simulaciones manuales")
    
    def create_detailed_comparison_table(self):
        """Crea tabla detallada de comparación"""
        logging.info("=== CREANDO TABLA DE COMPARACIÓN DETALLADA ===")
        
        # Crear DataFrame con todos los datos
        data = []
        
        # Agregar datos Brian2
        for i, (step, t, v) in enumerate(self.float64_data):
            row = {
                'step': step,
                'time_ms': t,
                'v_float64_brian': v,
                'v_float16_brian': self.float16_data[i][2] if i < len(self.float16_data) else np.nan,
                'v_posit16_brian': self.posit16_data[i][2] if i < len(self.posit16_data) else np.nan
            }
            data.append(row)
        
        # Agregar datos manuales
        for i, (step, t, v) in enumerate(self.manual_float64_data):
            if i < len(data):
                data[i].update({
                    'v_float64_manual': v,
                    'v_float16_manual': self.manual_float16_data[i][2] if i < len(self.manual_float16_data) else np.nan,
                    'v_posit16_manual': self.manual_posit16_data[i][2] if i < len(self.manual_posit16_data) else np.nan
                })
        
        df = pd.DataFrame(data)
        
        # Calcular diferencias
        if 'v_float64_brian' in df.columns and 'v_float64_manual' in df.columns:
            df['diff_brian_vs_manual_float64'] = df['v_float64_brian'] - df['v_float64_manual']
        
        if 'v_float16_brian' in df.columns and 'v_float16_manual' in df.columns:
            df['diff_brian_vs_manual_float16'] = df['v_float16_brian'] - df['v_float16_manual']
        
        # Guardar tabla
        os.makedirs('results/debug_numerical', exist_ok=True)
        df.to_csv('results/debug_numerical/detailed_comparison_table.csv', index=False)
        
        # Imprimir primeros 20 pasos
        print("\n=== TABLA DE COMPARACIÓN (PRIMEROS 20 PASOS) ===")
        print(df.head(20).to_string(index=False, float_format='%.6f'))
        
        return df
    
    def plot_comparison_analysis(self):
        """Genera gráficas de análisis comparativo"""
        logging.info("=== GENERANDO GRÁFICAS DE ANÁLISIS ===")
        
        os.makedirs('results/debug_numerical', exist_ok=True)
        
        # Gráfica 1: Comparación de voltajes
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Brian2
        plt.subplot(2, 2, 1)
        if self.float64_data:
            times = [d[1] for d in self.float64_data]
            v64 = [d[2] for d in self.float64_data]
            plt.plot(times, v64, 'k-', label='float64', linewidth=2)
        
        if self.float16_data:
            times = [d[1] for d in self.float16_data]
            v16 = [d[2] for d in self.float16_data]
            plt.plot(times, v16, 'b-', label='float16', linewidth=1, alpha=0.7)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (mV)')
        plt.title('Brian2: Comparación de Voltajes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Manual
        plt.subplot(2, 2, 2)
        if self.manual_float64_data:
            times = [d[1] for d in self.manual_float64_data]
            v64 = [d[2] for d in self.manual_float64_data]
            plt.plot(times, v64, 'k-', label='float64', linewidth=2)
        
        if self.manual_float16_data:
            times = [d[1] for d in self.manual_float16_data]
            v16 = [d[2] for d in self.manual_float16_data]
            plt.plot(times, v16, 'b-', label='float16', linewidth=1, alpha=0.7)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (mV)')
        plt.title('Manual: Comparación de Voltajes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Diferencias
        plt.subplot(2, 2, 3)
        if self.float64_data and self.float16_data:
            min_len = min(len(self.float64_data), len(self.float16_data))
            times = [self.float64_data[i][1] for i in range(min_len)]
            diff = [self.float64_data[i][2] - self.float16_data[i][2] for i in range(min_len)]
            plt.plot(times, diff, 'r-', label='float64 - float16', linewidth=1)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Diferencia (mV)')
        plt.title('Diferencias Brian2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Diferencias manuales
        plt.subplot(2, 2, 4)
        if self.manual_float64_data and self.manual_float16_data:
            min_len = min(len(self.manual_float64_data), len(self.manual_float16_data))
            times = [self.manual_float64_data[i][1] for i in range(min_len)]
            diff = [self.manual_float64_data[i][2] - self.manual_float16_data[i][2] for i in range(min_len)]
            plt.plot(times, diff, 'r-', label='float64 - float16', linewidth=1)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Diferencia (mV)')
        plt.title('Diferencias Manuales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/debug_numerical/comparison_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Gráfica 2: Análisis de errores acumulados
        plt.figure(figsize=(12, 8))
        
        if self.float64_data and self.float16_data:
            min_len = min(len(self.float64_data), len(self.float16_data))
            times = [self.float64_data[i][1] for i in range(min_len)]
            
            # Error absoluto
            abs_error = [abs(self.float64_data[i][2] - self.float16_data[i][2]) for i in range(min_len)]
            plt.subplot(2, 2, 1)
            plt.plot(times, abs_error, 'r-', label='Error absoluto')
            plt.xlabel('Tiempo (ms)')
            plt.ylabel('Error absoluto (mV)')
            plt.title('Error Absoluto Brian2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error acumulado
            cum_error = np.cumsum(abs_error)
            plt.subplot(2, 2, 2)
            plt.plot(times, cum_error, 'g-', label='Error acumulado')
            plt.xlabel('Tiempo (ms)')
            plt.ylabel('Error acumulado (mV)')
            plt.title('Error Acumulado Brian2')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if self.manual_float64_data and self.manual_float16_data:
            min_len = min(len(self.manual_float64_data), len(self.manual_float16_data))
            times = [self.manual_float64_data[i][1] for i in range(min_len)]
            
            # Error absoluto manual
            abs_error = [abs(self.manual_float64_data[i][2] - self.manual_float16_data[i][2]) for i in range(min_len)]
            plt.subplot(2, 2, 3)
            plt.plot(times, abs_error, 'r-', label='Error absoluto')
            plt.xlabel('Tiempo (ms)')
            plt.ylabel('Error absoluto (mV)')
            plt.title('Error Absoluto Manual')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Error acumulado manual
            cum_error = np.cumsum(abs_error)
            plt.subplot(2, 2, 4)
            plt.plot(times, cum_error, 'g-', label='Error acumulado')
            plt.xlabel('Tiempo (ms)')
            plt.ylabel('Error acumulado (mV)')
            plt.title('Error Acumulado Manual')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/debug_numerical/error_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo de depuración numérica"""
        logging.info("=== INICIANDO ANÁLISIS COMPLETO DE DEPURACIÓN NUMÉRICA ===")
        
        # 1. Simulación Brian2 con logging
        mon64, spk64, mon16, spk16 = self.run_brian2_simulation_with_logging()
        
        # 2. Simulación manual con NumPy
        spikes_float64, spikes_float16, spikes_posit16 = self.run_manual_numpy_simulation()
        
        # 3. Detección de divergencias
        self.detect_divergence_points()
        
        # 4. Crear tabla de comparación
        df = self.create_detailed_comparison_table()
        
        # 5. Generar gráficas
        self.plot_comparison_analysis()
        
        # 6. Resumen final
        logging.info("=== RESUMEN FINAL ===")
        logging.info(f"Simulación completada con {self.n_steps} pasos")
        logging.info(f"Archivos generados en results/debug_numerical/")
        logging.info(f"Log detallado guardado en debug_numerical.log")
        
        return df

def main():
    """Función principal"""
    print("=== DEPURACIÓN NUMÉRICA PASO A PASO ===")
    print("Implementando técnicas de depuración numérica avanzadas")
    print("1. Logging detallado paso a paso")
    print("2. Detección de puntos de divergencia")
    print("3. Simulación manual con NumPy")
    print("4. Callbacks durante la simulación")
    print("5. Análisis comparativo completo")
    print()
    
    # Crear instancia del debugger
    debugger = NumericalDebugger(dt=0.1*b2.ms, simulation_time=50*b2.ms)
    
    # Ejecutar análisis completo
    results_df = debugger.run_complete_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- debug_numerical.log: Log detallado")
    print("- results/debug_numerical/detailed_comparison_table.csv: Tabla de comparación")
    print("- results/debug_numerical/comparison_analysis.png: Gráficas de comparación")
    print("- results/debug_numerical/error_analysis.png: Análisis de errores")

if __name__ == "__main__":
    main() 