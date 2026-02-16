import json
import numpy as np
import os

class RegretExporter:
    """
    Exporta los resultados del RegretTracker a CSV o JSON.

    Esta clase permite mantener un formato uniforme de salida
    para comparar distintas polÃ­ticas o algoritmos.
    """

    @staticmethod
    def export_csv(tracker, args, ruta):
        """
        Exporta un tracker a CSV (resumen + series completas).

        ParÃ¡metros
        ----------
        tracker : RegretTracker
            Instancia con los datos del experimento.
        args : argparse.Namespace
            ParÃ¡metros del experimento.
        ruta : str
            Carpeta donde guardar los ficheros.
        """

        nombre = f"{tracker.name}_{args.batch_size}_{args.slate}_{args.epsilon}_{args.min_review_count}"
        fichero = os.path.join(ruta, nombre)

        # Resumen
        resumen = [
            "batch_size, slate_size, epsilon, min_reviews, final_regret, final_regret_estimado, final_regret_autoref",
            f"{args.batch_size}, {args.slate}, {args.epsilon}, {args.min_review_count}, "
            f"{tracker.cum_teorico[-1]}, {tracker.cum_estimado[-1]}, {tracker.cum_autoref[-1]}"
        ]

        with open(fichero + ".csv", "w") as f:
            for linea in resumen:
                f.write(linea + "\n")

        # Raw
        raw = {
            "regret_teorico": tracker.regret_teorico,
            "regret_estimado": tracker.regret_estimado,
            "regret_autoref": tracker.regret_autoref,
            "cum_teorico": tracker.cum_teorico.tolist(),
            "cum_estimado": tracker.cum_estimado.tolist(),
            "cum_autoref": tracker.cum_autoref.tolist()
        }

        with open(fichero + "_raw.json", "w") as f:
            json.dump(raw, f, indent=2)
