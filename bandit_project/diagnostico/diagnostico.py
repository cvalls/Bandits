from metricas.metricas import MetricaMejorBrazoFinal

class Diagnostico:
    """
    Ejecuta un conjunto de métricas sobre el estado final del experimento.
    Cada métrica es una instancia de una subclase de Metrica.
    """

    def __init__(self, metricas):
        self.metricas = metricas

    def ejecutar(self, contexto):
        """
        Ejecuta todas las métricas y devuelve un diccionario con los resultados.

        contexto: dict ya construido (viene de ResultadoExperimento.contexto_diagnostico)
        Debe contener al menos:
          - df
          - brazos
          - historicoRecomendados
          - historicoMeansPorBatch
          - sz_avg_reward_acum
          - slate
          - window_best
          - window_global
          - margen
        """
        resultados = {}

        # 1. Detectar si está la métrica del mejor brazo
        metrica_mejor_brazo = next(
            (m for m in self.metricas if isinstance(m, MetricaMejorBrazoFinal)),
            None
        )

        if metrica_mejor_brazo is not None:
            resultado_mejor_brazo = metrica_mejor_brazo.calcular(contexto)
            contexto["idMejorBrazo"] = resultado_mejor_brazo["best_arm_id"]
            resultados[metrica_mejor_brazo.nombre] = resultado_mejor_brazo

        # 2. Ejecutar el resto de métricas
        for metrica in self.metricas:
            if isinstance(metrica, MetricaMejorBrazoFinal):
                continue

            resultado = metrica.calcular(contexto)
            resultados[metrica.nombre] = resultado

        return resultados

    def imprimirMetricas(self, resultados):
        for nombre_metrica, resultados_metrica in resultados.items():
            print(f"\n=== {nombre_metrica.upper()} ===")
            for clave, (texto, valor) in resultados_metrica.items():
                print(f"{texto}: {valor}")
