import numpy as np
import pandas as pd
from typing import List, Tuple

# =============================================================================
# PROYECTO 2: CADENA DE MARKOV DE 4 ESTADOS
# Análisis del comportamiento de una máquina industrial
# =============================================================================

class CadenaMarkov4Estados:
    """
    Clase para simular y analizar una Cadena de Markov de 4 estados
    que representa el comportamiento de una máquina industrial.
    """
    
    def __init__(self):
        """
        Inicializa la cadena de Markov con la matriz de transición y estados.
        
        Estados:
        1 - Operativa
        2 - En mantenimiento preventivo
        3 - En reparación
        4 - Fuera de servicio
        """
        # Matriz de transición P
        self.P = np.array([
            [0.8, 0.1, 0.1, 0.0],  # Desde Operativa
            [0.6, 0.3, 0.1, 0.0],  # Desde Mantenimiento preventivo
            [0.2, 0.3, 0.4, 0.1],  # Desde Reparación
            [0.0, 0.1, 0.4, 0.5]   # Desde Fuera de servicio
        ])
        
        # Estado inicial (día 0): máquina operativa
        self.estado_inicial = np.array([1, 0, 0, 0])
        
        # Nombres de los estados
        self.nombres_estados = [
            "Operativa",
            "En mantenimiento preventivo",
            "En reparación",
            "Fuera de servicio"
        ]
    
    def simular_paso_a_paso(self, dias: int = 10) -> pd.DataFrame:
        """
        Simula la evolución de la cadena de Markov día a día.
        
        Args:
            dias: Número de días a simular (por defecto 10)
            
        Returns:
            DataFrame con las probabilidades de cada estado por día
        """
        print("=" * 80)
        print("ACTIVIDAD 1: SIMULACIÓN PASO A PASO (DÍA 1 AL 10)")
        print("=" * 80)
        
        # Almacenar resultados
        resultados = []
        estado_actual = self.estado_inicial.copy()
        
        # Día 0 (estado inicial)
        resultados.append({
            'Día': 0,
            'Operativa': estado_actual[0],
            'Mantenimiento': estado_actual[1],
            'Reparación': estado_actual[2],
            'Fuera de servicio': estado_actual[3]
        })
        
        print("\n📊 CÁLCULO MANUAL DE LOS PRIMEROS 3 PASOS:\n")
        
        # Simular cada día
        for dia in range(1, dias + 1):
            # Calcular siguiente estado: π(n) = π(n-1) * P
            estado_nuevo = estado_actual @ self.P
            
            # Mostrar cálculo detallado para los primeros 3 días
            if dia <= 3:
                print(f"Día {dia}:")
                print(f"  π({dia}) = π({dia-1}) × P")
                print(f"  π({dia}) = {estado_actual} × P")
                print(f"  π({dia}) = {estado_nuevo}")
                print(f"  Probabilidades:")
                for i, nombre in enumerate(self.nombres_estados):
                    print(f"    {nombre}: {estado_nuevo[i]:.4f} ({estado_nuevo[i]*100:.2f}%)")
                print()
            
            resultados.append({
                'Día': dia,
                'Operativa': estado_nuevo[0],
                'Mantenimiento': estado_nuevo[1],
                'Reparación': estado_nuevo[2],
                'Fuera de servicio': estado_nuevo[3]
            })
            
            estado_actual = estado_nuevo.copy()
        
        df_resultados = pd.DataFrame(resultados)
        
        print("\n📋 TABLA COMPLETA DE SIMULACIÓN (DÍA 0 AL 10):")
        print("-" * 80)
        print(df_resultados.to_string(index=False, float_format='%.6f'))
        print()
        
        return df_resultados
    
    def analizar_evolucion(self, df_resultados: pd.DataFrame) -> None:
        """
        Analiza la evolución de las probabilidades a lo largo del tiempo.
        
        Args:
            df_resultados: DataFrame con los resultados de la simulación
        """
        print("\n" + "=" * 80)
        print("ACTIVIDAD 2: ANÁLISIS DE EVOLUCIÓN")
        print("=" * 80)
        
        # Probabilidades en día 5
        dia_5 = df_resultados[df_resultados['Día'] == 5].iloc[0]
        print("\n📌 PROBABILIDADES AL DÍA 5:")
        print(f"  • Operativa: {dia_5['Operativa']:.6f} ({dia_5['Operativa']*100:.2f}%)")
        print(f"  • Mantenimiento: {dia_5['Mantenimiento']:.6f} ({dia_5['Mantenimiento']*100:.2f}%)")
        print(f"  • Reparación: {dia_5['Reparación']:.6f} ({dia_5['Reparación']*100:.2f}%)")
        print(f"  • Fuera de servicio: {dia_5['Fuera de servicio']:.6f} ({dia_5['Fuera de servicio']*100:.2f}%)")
        
        # Probabilidades en día 10
        dia_10 = df_resultados[df_resultados['Día'] == 10].iloc[0]
        print("\n📌 PROBABILIDADES AL DÍA 10:")
        print(f"  • Operativa: {dia_10['Operativa']:.6f} ({dia_10['Operativa']*100:.2f}%)")
        print(f"  • Mantenimiento: {dia_10['Mantenimiento']:.6f} ({dia_10['Mantenimiento']*100:.2f}%)")
        print(f"  • Reparación: {dia_10['Reparación']:.6f} ({dia_10['Reparación']*100:.2f}%)")
        print(f"  • Fuera de servicio: {dia_10['Fuera de servicio']:.6f} ({dia_10['Fuera de servicio']*100:.2f}%)")
        
        # Análisis de patrón
        print("\n🔍 ANÁLISIS DE PATRÓN:")
        
        # Cambios entre día 5 y 10
        cambio_operativa = dia_10['Operativa'] - dia_5['Operativa']
        cambio_mantenimiento = dia_10['Mantenimiento'] - dia_5['Mantenimiento']
        cambio_reparacion = dia_10['Reparación'] - dia_5['Reparación']
        cambio_fuera = dia_10['Fuera de servicio'] - dia_5['Fuera de servicio']
        
        print(f"  Cambio en probabilidad Operativa (día 5 a 10): {cambio_operativa:+.6f}")
        print(f"  Cambio en probabilidad Mantenimiento (día 5 a 10): {cambio_mantenimiento:+.6f}")
        print(f"  Cambio en probabilidad Reparación (día 5 a 10): {cambio_reparacion:+.6f}")
        print(f"  Cambio en probabilidad Fuera de servicio (día 5 a 10): {cambio_fuera:+.6f}")
        
        print("\n  Observación:")
        if abs(cambio_operativa) < 0.001:
            print("  ✓ Las probabilidades se están estabilizando (convergiendo al estado estacionario)")
        else:
            print("  ⚠ Las probabilidades aún están cambiando significativamente")
    
    def calcular_estado_estacionario(self) -> np.ndarray:
        """
        Calcula el estado estacionario resolviendo el sistema: π = πP y Σπ = 1
        
        Returns:
            Vector de probabilidades del estado estacionario
        """
        # Resolver (P^T - I)π = 0, sujeto a Σπ = 1
        n = len(self.P)
        
        # Crear sistema de ecuaciones
        A = self.P.T - np.eye(n)
        # Reemplazar última ecuación con la restricción de suma
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        
        # Resolver sistema
        pi_estacionario = np.linalg.solve(A, b)
        
        return pi_estacionario
    
    def interpretar_resultados(self, pi_estacionario: np.ndarray) -> None:
        """
        Interpreta los resultados y responde las preguntas del proyecto.
        
        Args:
            pi_estacionario: Vector de probabilidades del estado estacionario
        """
        print("\n" + "=" * 80)
        print("ACTIVIDAD 3: INTERPRETACIÓN DE RESULTADOS")
        print("=" * 80)
        
        print("\n🎯 ESTADO ESTACIONARIO (Probabilidades a largo plazo):")
        print("-" * 80)
        for i, nombre in enumerate(self.nombres_estados):
            print(f"  {nombre}: {pi_estacionario[i]:.6f} ({pi_estacionario[i]*100:.2f}%)")
        
        print("\n\n❓ PREGUNTA 1: ¿Cuál es la probabilidad a largo plazo de que la máquina esté operativa?")
        print(f"  📍 RESPUESTA: {pi_estacionario[0]:.6f} o {pi_estacionario[0]*100:.2f}%")
        print(f"     A largo plazo, la máquina estará operativa aproximadamente {pi_estacionario[0]*100:.1f}% del tiempo.")
        
        # Calcular confiabilidad del sistema
        prob_funcionamiento = pi_estacionario[0] + pi_estacionario[1]  # Operativa + Mantenimiento
        
        print("\n\n❓ PREGUNTA 2: ¿Qué tan confiable es el sistema?")
        print(f"  📍 RESPUESTA: La confiabilidad del sistema es {prob_funcionamiento:.6f} o {prob_funcionamiento*100:.2f}%")
        print(f"     (Estados funcionales: Operativa + Mantenimiento preventivo)")
        print(f"     Desglose:")
        print(f"       • Operativa: {pi_estacionario[0]*100:.2f}%")
        print(f"       • Mantenimiento preventivo: {pi_estacionario[1]*100:.2f}%")
        print(f"       • En reparación: {pi_estacionario[2]*100:.2f}%")
        print(f"       • Fuera de servicio: {pi_estacionario[3]*100:.2f}%")
        
        if prob_funcionamiento >= 0.85:
            evaluacion = "EXCELENTE"
        elif prob_funcionamiento >= 0.70:
            evaluacion = "BUENA"
        elif prob_funcionamiento >= 0.50:
            evaluacion = "ACEPTABLE"
        else:
            evaluacion = "DEFICIENTE"
        
        print(f"\n     Evaluación: {evaluacion}")
        
        print("\n\n❓ PREGUNTA 3: ¿Qué mejoras sugerirías al modelo?")
        print("  📍 SUGERENCIAS DE MEJORA:")
        print()
        
        # Analizar la matriz de transición para sugerencias
        if pi_estacionario[2] + pi_estacionario[3] > 0.20:
            print("  1. 🔧 AUMENTAR FRECUENCIA DE MANTENIMIENTO PREVENTIVO:")
            print("     • Actualmente, hay una probabilidad significativa de reparación y fuera de servicio")
            print("     • Aumentar P[0→1] (Operativa → Mantenimiento) de 0.1 a 0.15")
            print("     • Reducir P[0→2] (Operativa → Reparación) de 0.1 a 0.05")
            print("     • Esto reduciría las fallas catastróficas")
            print()
        
        print("  2. 📈 MEJORAR EFICIENCIA DEL MANTENIMIENTO:")
        print("     • Aumentar P[1→0] (Mantenimiento → Operativa) de 0.6 a 0.75")
        print("     • Reducir P[1→2] (Mantenimiento → Reparación) de 0.1 a 0.05")
        print("     • Capacitar mejor al personal de mantenimiento")
        print()
        
        print("  3. 🚀 MEJORAR CAPACIDAD DE REPARACIÓN:")
        print("     • Aumentar P[2→0] (Reparación → Operativa) de 0.2 a 0.4")
        print("     • Reducir P[2→3] (Reparación → Fuera de servicio) de 0.1 a 0.05")
        print("     • Tener repuestos críticos en stock")
        print()
        
        print("  4. 💰 ANÁLISIS COSTO-BENEFICIO:")
        print("     • Comparar el costo del mantenimiento preventivo adicional")
        print("     • versus el costo de tiempo fuera de servicio y reparaciones mayores")
        print()
    
    def simular_escenario_mejorado(self) -> None:
        """
        Simula un escenario con las mejoras sugeridas para comparar.
        """
        print("\n" + "=" * 80)
        print("SIMULACIÓN DE ESCENARIO MEJORADO")
        print("=" * 80)
        
        # Matriz mejorada con las sugerencias
        P_mejorada = np.array([
            [0.70, 0.15, 0.05, 0.10],  # Más mantenimiento preventivo
            [0.75, 0.20, 0.05, 0.00],  # Mejor retorno a operativa
            [0.40, 0.30, 0.25, 0.05],  # Mejor capacidad de reparación
            [0.00, 0.20, 0.30, 0.50]   # Fuera de servicio
        ])
        
        # Calcular estado estacionario mejorado
        n = len(P_mejorada)
        A = P_mejorada.T - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        pi_mejorado = np.linalg.solve(A, b)
        
        print("\n📊 COMPARACIÓN: Estado Actual vs. Mejorado")
        print("-" * 80)
        print(f"{'Estado':<30} {'Actual':<15} {'Mejorado':<15} {'Cambio':<15}")
        print("-" * 80)
        
        pi_actual = self.calcular_estado_estacionario()
        
        for i, nombre in enumerate(self.nombres_estados):
            cambio = pi_mejorado[i] - pi_actual[i]
            print(f"{nombre:<30} {pi_actual[i]*100:>6.2f}%        {pi_mejorado[i]*100:>6.2f}%        {cambio*100:>+6.2f}%")
        
        print("-" * 80)
        prob_func_actual = pi_actual[0] + pi_actual[1]
        prob_func_mejorado = pi_mejorado[0] + pi_mejorado[1]
        print(f"{'CONFIABILIDAD TOTAL':<30} {prob_func_actual*100:>6.2f}%        {prob_func_mejorado*100:>6.2f}%        {(prob_func_mejorado-prob_func_actual)*100:>+6.2f}%")
        print()
    
    def ejecutar_analisis_completo(self) -> None:
        """
        Ejecuta el análisis completo del proyecto.
        """
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "    PROYECTO 2: CADENA DE MARKOV DE 4 ESTADOS".center(78) + "║")
        print("║" + "    Análisis del Comportamiento de una Máquina Industrial".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        
        print("📋 INFORMACIÓN DEL SISTEMA:")
        print("-" * 80)
        print("Estados posibles:")
        for i, nombre in enumerate(self.nombres_estados, 1):
            print(f"  {i}. {nombre}")
        print()
        print("Matriz de Transición P:")
        print(self.P)
        print()
        print("Estado Inicial (Día 0):")
        print(f"  π(0) = {self.estado_inicial}  [Máquina operativa]")
        print()
        
        # 1. Simulación paso a paso
        df_resultados = self.simular_paso_a_paso(dias=10)
        
        # 2. Análisis de evolución
        self.analizar_evolucion(df_resultados)
        
        # 3. Cálculo del estado estacionario
        pi_estacionario = self.calcular_estado_estacionario()
        
        # 4. Interpretación de resultados
        self.interpretar_resultados(pi_estacionario)
        
        # 5. Simulación de escenario mejorado
        self.simular_escenario_mejorado()
        
        print("\n" + "=" * 80)
        print("✅ ANÁLISIS COMPLETADO")
        print("=" * 80)
        print()


# =============================================================================
# FUNCIÓN PRINCIPAL 
# =============================================================================

def main():
    """
    Función principal que ejecuta el proyecto completo.
    """
    # Crear instancia de la cadena de Markov
    cadena = CadenaMarkov4Estados()
    
    # Ejecutar análisis completo
    cadena.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()