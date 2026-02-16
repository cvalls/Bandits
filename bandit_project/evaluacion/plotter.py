import matplotlib.pyplot as plt
from .metrics import moving_average

class RegretPlotter:
    """
    Herramientas de visualizaciÃ³n para regrets.
    """

    @staticmethod
    def plot_regrets(tracker, window=1000):
        """
        Grafica regret instantÃ¡neo y acumulado para un tracker.

        ParÃ¡metros
        ----------
        tracker : RegretTracker
            Instancia con los datos del experimento.
        window : int
            Ventana para la media mÃ³vil.
        """

        plt.figure(figsize=(14, 5))

        # Regret instantÃ¡neo suavizado
        plt.subplot(1, 2, 1)
        plt.plot(moving_average(tracker.regret_teorico, window), label="TeÃ³rico")
        plt.plot(moving_average(tracker.regret_estimado, window), label="Estimado")
        plt.plot(moving_average(tracker.regret_autoref, window), label="Auto-referencial")
        plt.title(f"Regret instantÃ¡neo ({tracker.name})")
        plt.xlabel("Batch")
        plt.ylabel("Regret")
        plt.legend()

        # Regret acumulado
        plt.subplot(1, 2, 2)
        plt.plot(tracker.cum_teorico, label="TeÃ³rico")
        plt.plot(tracker.cum_estimado, label="Estimado")
        plt.plot(tracker.cum_autoref, label="Auto-referencial")
        plt.title(f"Regret acumulado ({tracker.name})")
        plt.xlabel("Batch")
        plt.ylabel("Regret acumulado")
        plt.legend()

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_regrets_comparison(trackers, labels=None, tipo="teorico"):
        """
        Grafica en una sola figura el regret acumulado de varias polÃ­ticas.

        ParÃ¡metros
        ----------
        trackers : list[RegretTracker]
            Lista de instancias de RegretTracker (una por polÃ­tica).
        labels : list[str], opcional
            Etiquetas para cada polÃ­tica. Si no se dan, se usan los nombres del tracker.
        tipo : str
            Tipo de regret a graficar: "teorico", "estimado" o "autoref".

        Ejemplo
        -------
        RegretPlotter.plot_regrets_comparison(
            [reg_fijo, reg_decay, reg_var],
            ["Fijo", "Decay", "Var"],
            tipo="teorico"
        )
        """

        if labels is None:
            labels = [tracker.name for tracker in trackers]

        # SelecciÃ³n del atributo segÃºn tipo
        if tipo == "teorico":
            attr = "cum_teorico"
            titulo = "Regret acumulado teÃ³rico"
        elif tipo == "estimado":
            attr = "cum_estimado"
            titulo = "Regret acumulado estimado"
        elif tipo == "autoref":
            attr = "cum_autoref"
            titulo = "Regret acumulado auto-referencial"
        else:
            raise ValueError("tipo debe ser 'teorico', 'estimado' o 'autoref'")

        plt.figure(figsize=(12, 6))

        for tracker, label in zip(trackers, labels):
            serie = getattr(tracker, attr)
            if serie is None:
                raise ValueError(f"El tracker '{tracker.name}' no tiene acumulados. "
                                 f"Â¿Has llamado a compute_cumulative()?")

            plt.plot(serie, label=label)

        plt.title(titulo)
        plt.xlabel("Batch")
        plt.ylabel("Regret acumulado")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

