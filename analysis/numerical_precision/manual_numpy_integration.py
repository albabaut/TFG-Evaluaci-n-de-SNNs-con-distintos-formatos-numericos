import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from posit_wrapper import convert16
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('manual_numpy_integration.log'),
        logging.StreamHandler()
    ]
)

class ManualNumpyIntegration:
    def __init__(self, dt=0.1, simulation_time=100):
        """
        Inicializa el integrador manual con NumPy
        
        Args:
            dt: Paso de tiempo en ms
            simulation_time: Tiempo total de simulación en ms
        """
        self.dt = dt
        self.simulation_time = simulation_time
        self.n_steps = int(simulation_time / dt)
        
        # Parámetros de la neurona LIF
        self.tau = 10.0  # ms
        self.V_rest = -70.0  # mV
        self.Vth = -50.0  # mV
        self.Vreset = -70.0  # mV
        self.I_ext = 20.0  # mV
        
        # Almacenamiento de datos
        self.results = {
            'float64': {'voltage': [], 'time': [], 'spikes': []},
            'float32': {'voltage': [], 'time': [], 'spikes': []},
            'float16': {'voltage': [], 'time': [], 'spikes': []},
            'posit16': {'voltage': [], 'time': [], 'spikes': []}
        }
        
        logging.info(f"ManualNumpyIntegration inicializado con {self.n_steps} pasos de {dt}ms")
    
    def integrate_euler_explicit(self, precision_type='float64'):
        """
        Integración explícita de Euler para LIF
        
        Args:
            precision_type: Tipo de precisión ('float64', 'float32', 'float16', 'posit16')
        """
        logging.info(f"Ejecutando integración Euler explícita con {precision_type}")
        
        # Inicializar variables según el tipo de precisión
        if precision_type == 'float64':
            v = np.float64(self.V_rest)
            tau = np.float64(self.tau)
            V_rest = np.float64(self.V_rest)
            Vth = np.float64(self.Vth)
            Vreset = np.float64(self.Vreset)
            I_ext = np.float64(self.I_ext)
            dt = np.float64(self.dt)
        elif precision_type == 'float32':
            v = np.float32(self.V_rest)
            tau = np.float32(self.tau)
            V_rest = np.float32(self.V_rest)
            Vth = np.float32(self.Vth)
            Vreset = np.float32(self.Vreset)
            I_ext = np.float32(self.I_ext)
            dt = np.float32(self.dt)
        elif precision_type == 'float16':
            v = np.float16(self.V_rest)
            tau = np.float16(self.tau)
            V_rest = np.float16(self.V_rest)
            Vth = np.float16(self.Vth)
            Vreset = np.float16(self.Vreset)
            I_ext = np.float16(self.I_ext)
            dt = np.float16(self.dt)
        elif precision_type == 'posit16':
            v = convert16(self.V_rest)
            tau = convert16(self.tau)
            V_rest = convert16(self.V_rest)
            Vth = convert16(self.Vth)
            Vreset = convert16(self.Vreset)
            I_ext = convert16(self.I_ext)
            dt = convert16(self.dt)
        
        voltage_history = []
        time_history = []
        spikes = []
        
        for step in range(self.n_steps):
            t = step * dt
            
            # Ecuación diferencial: dv/dt = (I_ext - (v - V_rest)) / tau
            # Método de Euler explícito: v(t+dt) = v(t) + dt * dv/dt
            dv_dt = (I_ext - (v - V_rest)) / tau
            
            # Aplicar conversión de precisión en cada operación
            if precision_type == 'float16':
                dv = np.float16(dv_dt * dt)
                v = np.float16(v + dv)
            elif precision_type == 'posit16':
                dv = convert16(dv_dt * dt)
                v = convert16(v + dv)
            else:
                v = v + dv_dt * dt
            
            # Detección de spike
            if v > Vth:
                spikes.append(t)
                if precision_type == 'float16':
                    v = np.float16(Vreset)
                elif precision_type == 'posit16':
                    v = convert16(Vreset)
                else:
                    v = Vreset
            
            # Almacenar datos
            voltage_history.append(float(v))
            time_history.append(float(t))
            
            # Log cada 100 pasos
            if step % 100 == 0:
                logging.debug(f"{precision_type} - Paso {step}: t={t:.3f}ms, v={float(v):.6f}mV")
        
        # Guardar resultados
        self.results[precision_type]['voltage'] = voltage_history
        self.results[precision_type]['time'] = time_history
        self.results[precision_type]['spikes'] = spikes
        
        logging.info(f"Integración {precision_type} completada. {len(spikes)} spikes detectados")
        
        return voltage_history, time_history, spikes
    
    def integrate_analytical(self, precision_type='float64'):
        """
        Integración analítica para LIF (solución exacta)
        
        Args:
            precision_type: Tipo de precisión
        """
        logging.info(f"Ejecutando integración analítica con {precision_type}")
        
        # Inicializar variables según el tipo de precisión
        if precision_type == 'float64':
            v = np.float64(self.V_rest)
            tau = np.float64(self.tau)
            V_rest = np.float64(self.V_rest)
            Vth = np.float64(self.Vth)
            Vreset = np.float64(self.V_reset)
            I_ext = np.float64(self.I_ext)
            dt = np.float64(self.dt)
        elif precision_type == 'float16':
            v = np.float16(self.V_rest)
            tau = np.float16(self.tau)
            V_rest = np.float16(self.V_rest)
            Vth = np.float16(self.Vth)
            Vreset = np.float16(self.Vreset)
            I_ext = np.float16(self.I_ext)
            dt = np.float16(self.dt)
        elif precision_type == 'posit16':
            v = convert16(self.V_rest)
            tau = convert16(self.tau)
            V_rest = convert16(self.V_rest)
            Vth = convert16(self.Vth)
            Vreset = convert16(self.Vreset)
            I_ext = convert16(self.I_ext)
            dt = convert16(self.dt)
        
        voltage_history = []
        time_history = []
        spikes = []
        
        # Solución analítica: v(t) = V_rest + I_ext * tau * (1 - exp(-t/tau))
        for step in range(self.n_steps):
            t = step * dt
            
            # Calcular voltaje analítico
            if precision_type == 'float16':
                exp_term = np.float16(np.exp(-t / tau))
                v_analytical = np.float16(V_rest + I_ext * tau * (np.float16(1.0) - exp_term))
            elif precision_type == 'posit16':
                exp_term = convert16(np.exp(-t / tau))
                v_analytical = convert16(V_rest + I_ext * tau * (convert16(1.0) - exp_term))
            else:
                exp_term = np.exp(-t / tau)
                v_analytical = V_rest + I_ext * tau * (1.0 - exp_term)
            
            # Detección de spike
            if v_analytical > Vth:
                spikes.append(t)
                if precision_type == 'float16':
                    v_analytical = np.float16(Vreset)
                elif precision_type == 'posit16':
                    v_analytical = convert16(Vreset)
                else:
                    v_analytical = Vreset
            
            voltage_history.append(float(v_analytical))
            time_history.append(float(t))
        
        # Guardar resultados
        self.results[precision_type]['voltage'] = voltage_history
        self.results[precision_type]['time'] = time_history
        self.results[precision_type]['spikes'] = spikes
        
        logging.info(f"Integración analítica {precision_type} completada. {len(spikes)} spikes detectados")
        
        return voltage_history, time_history, spikes
    
    def detect_divergence_points(self, tolerance=0.01):
        """
        Detecta puntos de divergencia entre diferentes precisiones
        
        Args:
            tolerance: Tolerancia para considerar divergencia (mV)
        """
        logging.info("=== DETECTANDO PUNTOS DE DIVERGENCIA ===")
        
        divergence_points = {}
        
        # Comparar float64 vs otros tipos
        reference_data = self.results['float64']['voltage']
        reference_time = self.results['float64']['time']
        
        for precision_type in ['float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                test_data = self.results[precision_type]['voltage']
                min_len = min(len(reference_data), len(test_data))
                
                # Calcular diferencias
                diff = np.abs(np.array(reference_data[:min_len]) - np.array(test_data[:min_len]))
                
                # Encontrar primer punto de divergencia
                divergence_idx = np.argmax(diff > tolerance)
                
                if diff[divergence_idx] > tolerance:
                    divergence_points[precision_type] = {
                        'step': divergence_idx,
                        'time_ms': reference_time[divergence_idx],
                        'difference_mV': diff[divergence_idx],
                        'v_float64': reference_data[divergence_idx],
                        'v_test': test_data[divergence_idx]
                    }
                    
                    logging.warning(f"Divergencia {precision_type} vs float64 detectada:")
                    logging.warning(f"  Paso: {divergence_idx}")
                    logging.warning(f"  Tiempo: {reference_time[divergence_idx]:.3f}ms")
                    logging.warning(f"  Diferencia: {diff[divergence_idx]:.6f}mV")
                else:
                    logging.info(f"No se detectaron divergencias significativas para {precision_type}")
        
        return divergence_points
    
    def create_comparison_table(self):
        """Crea tabla de comparación detallada"""
        logging.info("=== CREANDO TABLA DE COMPARACIÓN ===")
        
        # Crear DataFrame con todos los datos
        data = []
        
        # Usar float64 como referencia
        reference_data = self.results['float64']
        n_steps = len(reference_data['voltage'])
        
        for i in range(n_steps):
            row = {
                'step': i,
                'time_ms': reference_data['time'][i],
                'v_float64': reference_data['voltage'][i]
            }
            
            # Agregar datos de otras precisiones
            for precision_type in ['float32', 'float16', 'posit16']:
                if i < len(self.results[precision_type]['voltage']):
                    row[f'v_{precision_type}'] = self.results[precision_type]['voltage'][i]
                    row[f'diff_{precision_type}'] = abs(reference_data['voltage'][i] - self.results[precision_type]['voltage'][i])
                else:
                    row[f'v_{precision_type}'] = np.nan
                    row[f'diff_{precision_type}'] = np.nan
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Guardar tabla
        os.makedirs('results/manual_numpy', exist_ok=True)
        df.to_csv('results/manual_numpy/comparison_table.csv', index=False)
        
        # Imprimir primeros 20 pasos
        print("\n=== TABLA DE COMPARACIÓN (PRIMEROS 20 PASOS) ===")
        print(df.head(20).to_string(index=False, float_format='%.6f'))
        
        return df
    
    def plot_comparison_analysis(self):
        """Genera gráficas de análisis comparativo"""
        logging.info("=== GENERANDO GRÁFICAS DE ANÁLISIS ===")
        
        os.makedirs('results/manual_numpy', exist_ok=True)
        
        # Gráfica 1: Comparación de voltajes
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Voltajes
        plt.subplot(2, 3, 1)
        for precision_type in ['float64', 'float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                plt.plot(self.results[precision_type]['time'], 
                        self.results[precision_type]['voltage'], 
                        label=precision_type, linewidth=2 if precision_type == 'float64' else 1)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Voltaje (mV)')
        plt.title('Comparación de Voltajes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Diferencias vs float64
        plt.subplot(2, 3, 2)
        reference_data = self.results['float64']
        for precision_type in ['float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                min_len = min(len(reference_data['voltage']), len(self.results[precision_type]['voltage']))
                diff = np.array(reference_data['voltage'][:min_len]) - np.array(self.results[precision_type]['voltage'][:min_len])
                plt.plot(reference_data['time'][:min_len], diff, label=f'float64 - {precision_type}')
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Diferencia (mV)')
        plt.title('Diferencias vs float64')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Error absoluto
        plt.subplot(2, 3, 3)
        for precision_type in ['float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                min_len = min(len(reference_data['voltage']), len(self.results[precision_type]['voltage']))
                abs_error = np.abs(np.array(reference_data['voltage'][:min_len]) - np.array(self.results[precision_type]['voltage'][:min_len]))
                plt.plot(reference_data['time'][:min_len], abs_error, label=f'Error {precision_type}')
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Error absoluto (mV)')
        plt.title('Error Absoluto')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Error acumulado
        plt.subplot(2, 3, 4)
        for precision_type in ['float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                min_len = min(len(reference_data['voltage']), len(self.results[precision_type]['voltage']))
                abs_error = np.abs(np.array(reference_data['voltage'][:min_len]) - np.array(self.results[precision_type]['voltage'][:min_len]))
                cum_error = np.cumsum(abs_error)
                plt.plot(reference_data['time'][:min_len], cum_error, label=f'Error acumulado {precision_type}')
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Error acumulado (mV)')
        plt.title('Error Acumulado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Histograma de diferencias
        plt.subplot(2, 3, 5)
        for precision_type in ['float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['voltage']) > 0:
                min_len = min(len(reference_data['voltage']), len(self.results[precision_type]['voltage']))
                diff = np.array(reference_data['voltage'][:min_len]) - np.array(self.results[precision_type]['voltage'][:min_len])
                plt.hist(diff, bins=50, alpha=0.7, label=precision_type)
        
        plt.xlabel('Diferencia (mV)')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Diferencias')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Spikes comparación
        plt.subplot(2, 3, 6)
        for precision_type in ['float64', 'float32', 'float16', 'posit16']:
            if len(self.results[precision_type]['spikes']) > 0:
                spikes = self.results[precision_type]['spikes']
                plt.vlines(spikes, 0, 1, label=f'{precision_type} ({len(spikes)} spikes)', alpha=0.7)
        
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Spikes')
        plt.title('Comparación de Spikes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/manual_numpy/comparison_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo"""
        logging.info("=== INICIANDO ANÁLISIS COMPLETO DE INTEGRACIÓN MANUAL ===")
        
        # 1. Integración Euler explícita para todas las precisiones
        for precision_type in ['float64', 'float32', 'float16', 'posit16']:
            self.integrate_euler_explicit(precision_type)
        
        # 2. Detección de divergencias
        divergence_points = self.detect_divergence_points()
        
        # 3. Crear tabla de comparación
        df = self.create_comparison_table()
        
        # 4. Generar gráficas
        self.plot_comparison_analysis()
        
        # 5. Resumen final
        logging.info("=== RESUMEN FINAL ===")
        for precision_type in ['float64', 'float32', 'float16', 'posit16']:
            n_spikes = len(self.results[precision_type]['spikes'])
            logging.info(f"{precision_type}: {n_spikes} spikes")
        
        logging.info(f"Archivos generados en results/manual_numpy/")
        logging.info(f"Log guardado en manual_numpy_integration.log")
        
        return df, divergence_points

def main():
    """Función principal"""
    print("=== INTEGRACIÓN MANUAL CON NUMPY ===")
    print("Implementando integración manual para comparar con Brian2")
    print("1. Integración Euler explícita")
    print("2. Múltiples precisiones numéricas")
    print("3. Detección de divergencias")
    print("4. Análisis comparativo")
    print()
    
    # Crear instancia del integrador
    integrator = ManualNumpyIntegration(dt=0.1, simulation_time=50)
    
    # Ejecutar análisis completo
    df, divergence_points = integrator.run_complete_analysis()
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa los archivos generados:")
    print("- manual_numpy_integration.log: Log detallado")
    print("- results/manual_numpy/comparison_table.csv: Tabla de comparación")
    print("- results/manual_numpy/comparison_analysis.png: Gráficas de análisis")

if __name__ == "__main__":
    main() 