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
        # Matriz de transición P (filas: estado actual, columnas: siguiente estado)
        self.P = np.array([
            [0.8, 0.1, 0.1, 0.0],  # Desde Operativa
            [0.6, 0.3, 0.1, 0.0],  # Desde Mantenimiento preventivo
            [0.2, 0.3, 0.4, 0.1],  # Desde Reparación
            [0.0, 0.1, 0.4, 0.5]   # Desde Fuera de servicio
        ], dtype=float)

        # Estado inicial (día 0): máquina operativa
        self.estado_inicial = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # Nombres de los estados
        self.nombres_estados = [
            "Operativa",
            "En mantenimiento preventivo",
            "En reparación",
            "Fuera de servicio"
        ]

        # Validaciones básicas
        self._validar_matriz()

    def _validar_matriz(self) -> None:
        """Valida que P sea estocástica por filas: no negativa y filas suman 1."""
        if (self.P < -1e-12).any():
            raise ValueError("La matriz P contiene probabilidades negativas.")
        row_sums = self.P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(
                f"Cada fila de P debe sumar 1. Sumas actuales: {row_sums}."
            )

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

        print("\nCÁLCULO MANUAL DE LOS PRIMEROS 3 PASOS:\n")

        # Simular cada día
        for dia in range(1, dias + 1):
            # Calcular siguiente estado: π(n) = π(n-1) * P
            estado_nuevo = estado_actual @ self.P

            # Mostrar cálculo detallado para los primeros 3 días
            if dia <= 3:
                print(f"Día {dia}:")
                print(f"  π({dia}) = π({dia-1}) × P")
                print(f"  π({dia}) = {np.round(estado_actual, 6)} × P")
                print(f"  π({dia}) = {np.round(estado_nuevo, 6)}")
                print("  Probabilidades:")
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

            estado_actual = estado_nuevo

        df_resultados = pd.DataFrame(resultados)

        print("\nTABLA COMPLETA DE SIMULACIÓN (DÍA 0 AL 10):")
        print("-" * 80)
        print(df_resultados.round(6).to_string(index=False))
        print()

        return df_resultados

    def _iterar_desde(self, pi0: np.ndarray, dias: int) -> np.ndarray:
        """Itera π(n+1)=π(n)·P 'dias' veces y devuelve el vector final."""
        estado = pi0.copy()
        for _ in range(dias):
            estado = estado @ self.P
        return estado

    def _dia_casi_convergencia(self, start_df: pd.DataFrame, max_dias: int = 50, tol: float = 1e-3) -> int | None:
        """
        Busca el primer día n (entre último del df y max_dias) tal que
        max(|π(n)-π(n-1)|) < tol. Devuelve el día o None si no se alcanza.
        """
        ultimo_dia = int(start_df['Día'].max())
        estado_prev = start_df.loc[start_df['Día'] == ultimo_dia, ['Operativa','Mantenimiento','Reparación','Fuera de servicio']].to_numpy().ravel()
        for dia in range(ultimo_dia + 1, max_dias + 1):
            estado_nuevo = estado_prev @ self.P
            if np.max(np.abs(estado_nuevo - estado_prev)) < tol:
                return dia
            estado_prev = estado_nuevo
        return None

    def analizar_evolucion(self, df_resultados: pd.DataFrame) -> None:
        """
        Analiza la evolución de las probabilidades a lo largo del tiempo.
        """
        print("\n" + "=" * 80)
        print("ACTIVIDAD 2: ANÁLISIS DE EVOLUCIÓN")
        print("=" * 80)

        # Probabilidades en día 5
        dia_5 = df_resultados.loc[df_resultados['Día'] == 5].iloc[0]
        print("\nPROBABILIDADES AL DÍA 5:")
        print(f"  • Operativa: {dia_5['Operativa']:.6f} ({dia_5['Operativa']*100:.2f}%)")
        print(f"  • Mantenimiento: {dia_5['Mantenimiento']:.6f} ({dia_5['Mantenimiento']*100:.2f}%)")
        print(f"  • Reparación: {dia_5['Reparación']:.6f} ({dia_5['Reparación']*100:.2f}%)")
        print(f"  • Fuera de servicio: {dia_5['Fuera de servicio']:.6f} ({dia_5['Fuera de servicio']*100:.2f}%)")

        # Probabilidades en día 10
        dia_10 = df_resultados.loc[df_resultados['Día'] == 10].iloc[0]
        print("\nPROBABILIDADES AL DÍA 10:")
        print(f"  • Operativa: {dia_10['Operativa']:.6f} ({dia_10['Operativa']*100:.2f}%)")
        print(f"  • Mantenimiento: {dia_10['Mantenimiento']:.6f} ({dia_10['Mantenimiento']*100:.2f}%)")
        print(f"  • Reparación: {dia_10['Reparación']:.6f} ({dia_10['Reparación']*100:.2f}%)")
        print(f"  • Fuera de servicio: {dia_10['Fuera de servicio']:.6f} ({dia_10['Fuera de servicio']*100:.2f}%)")

        # Análisis de patrón
        print("\nANÁLISIS DE PATRÓN:")

        # Cambios entre día 5 y 10
        cambio_operativa = dia_10['Operativa'] - dia_5['Operativa']
        cambio_mantenimiento = dia_10['Mantenimiento'] - dia_5['Mantenimiento']
        cambio_reparacion = dia_10['Reparación'] - dia_5['Reparación']
        cambio_fuera = dia_10['Fuera de servicio'] - dia_5['Fuera de servicio']

        print(f"  Cambio Operativa (5→10): {cambio_operativa:+.6f}")
        print(f"  Cambio Mantenimiento (5→10): {cambio_mantenimiento:+.6f}")
        print(f"  Cambio Reparación (5→10): {cambio_reparacion:+.6f}")
        print(f"  Cambio Fuera de servicio (5→10): {cambio_fuera:+.6f}")

        casi = self._dia_casi_convergencia(df_resultados, max_dias=50, tol=1e-3)
        print("\n  Observación:")
        if casi is None:
            print("  Aún hay cambios apreciables (para ver el largo plazo, revisar día 50).")
        else:
            print(f"  A partir del día {casi} los cambios son < 0.001 (casi convergencia).")

    def interpretar_resultados_por_iteracion(self, df_resultados: pd.DataFrame, dia_largo_plazo: int = 50) -> None:
        """
        Interpreta resultados SIN resolver el sistema: aproxima el largo plazo iterando hasta 'dia_largo_plazo'.
        """
        print("\n" + "=" * 80)
        print("ACTIVIDAD 3: INTERPRETACIÓN DE RESULTADOS (POR ITERACIÓN)")
        print("=" * 80)

        # Tomamos el último estado del df (día 10) y seguimos hasta el día 50
        ultimo_dia = int(df_resultados['Día'].max())
        estado_ult = df_resultados.loc[df_resultados['Día'] == ultimo_dia, ['Operativa','Mantenimiento','Reparación','Fuera de servicio']].to_numpy().ravel()
        pasos_extra = max(0, dia_largo_plazo - ultimo_dia)
        estado_dia50 = self._iterar_desde(estado_ult, pasos_extra)

        print(f"\nEstimación a largo plazo por iteración (día {dia_largo_plazo}):")
        print("-" * 80)
        for i, nombre in enumerate(self.nombres_estados):
            print(f"  {nombre}: {estado_dia50[i]:.6f} ({estado_dia50[i]*100:.2f}%)")

        # Confiabilidad: Operativa + Mantenimiento
        prob_funcionamiento = estado_dia50[0] + estado_dia50[1]

        print("\nPREGUNTA 1: ¿Cuál es la probabilidad a largo plazo de que la máquina esté operativa (por iteración)?")
        print(f"  RESPUESTA (día {dia_largo_plazo}): {estado_dia50[0]:.6f} ({estado_dia50[0]*100:.2f}%)")

        print("\nPREGUNTA 2: ¿Qué tan confiable es el sistema?")
        print(f"  RESPUESTA (día {dia_largo_plazo}): {prob_funcionamiento:.6f} ({prob_funcionamiento*100:.2f}%)")
        print("     (Estados funcionales: Operativa + Mantenimiento preventivo)")
        print("     Desglose:")
        print(f"       • Operativa: {estado_dia50[0]*100:.2f}%")
        print(f"       • Mantenimiento preventivo: {estado_dia50[1]*100:.2f}%")
        print(f"       • En reparación: {estado_dia50[2]*100:.2f}%")
        print(f"       • Fuera de servicio: {estado_dia50[3]*100:.2f}%")

        if prob_funcionamiento >= 0.85:
            evaluacion = "EXCELENTE"
        elif prob_funcionamiento >= 0.70:
            evaluacion = "BUENA"
        elif prob_funcionamiento >= 0.50:
            evaluacion = "ACEPTABLE"
        else:
            evaluacion = "DEFICIENTE"
        print(f"\n     Evaluación: {evaluacion}")

        print("\nPREGUNTA 3: ¿Qué mejoras sugerirías al modelo?")
        print("  SUGERENCIAS DE MEJORA (sin simular nuevo escenario):")
        print("  2) Mejorar eficiencia del Mantenimiento: subir P[Mantenimiento→Operativa] y bajar P[Mantenimiento→Reparación].")
        print("  3) Mejorar capacidad de Reparación: subir P[Reparación→Operativa] y bajar P[Reparación→FueraServicio].")
        print("  4) Análisis costo–beneficio de los cambios propuestos.")
        print()

    def ejecutar_analisis_completo(self) -> None:
        """
        Ejecuta el análisis completo del proyecto (solo lo pedido en el enunciado).
        """
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "    PROYECTO 2: CADENA DE MARKOV DE 4 ESTADOS".center(78) + "║")
        print("║" + "    Análisis del Comportamiento de una Máquina Industrial".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()

        print("INFORMACIÓN DEL SISTEMA:")
        print("-" * 80)
        print("Estados posibles:")
        for i, nombre in enumerate(self.nombres_estados, 1):
            print(f"  {i}. {nombre}")
        print("\nMatriz de Transición P:")
        print(self.P)
        print("\nEstado Inicial (Día 0):")
        print(f"  π(0) = {self.estado_inicial}  [Máquina operativa]\n")

        # 1. Simulación paso a paso
        df_resultados = self.simular_paso_a_paso(dias=10)

        # 2. Análisis de evolución
        self.analizar_evolucion(df_resultados)

        # 3. Interpretación por iteración (día 50 como “largo plazo”)
        self.interpretar_resultados_por_iteracion(df_resultados, dia_largo_plazo=50)

        print("\n" + "=" * 80)
        print("ANÁLISIS COMPLETADO")
        print("=" * 80)
        print()


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta el proyecto completo.
    """
    cadena = CadenaMarkov4Estados()
    cadena.ejecutar_analisis_completo()

if __name__ == "__main__":
    main()
