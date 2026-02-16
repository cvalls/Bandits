import numpy as np

class RegretTracker:
    """
    Clase encargada de gestionar el cÃ¡lculo de regret para una polÃ­tica concreta.

    Calcula y almacena:
    - Regret teÃ³rico: basado en las medias reales del entorno.
    - Regret estimado: basado en las medias estimadas por el bandido.
    - Regret auto-referencial: diferencia entre estimado y teÃ³rico.

    Esta clase es independiente del algoritmo de bandits. Solo requiere:
    - best_mean_real: valor real del mejor brazo.
    - true_means: vector de medias reales.
    - means_estimadas: vector de medias estimadas por el algoritmo.
    - arm_to_idx: diccionario movieId ? Ã­ndice.
    - emparejados: DataFrame con las pelÃ­culas realmente observadas en el batch.
    """

    def __init__(self, name: str):
        """
        Inicializa un tracker para una polÃ­tica concreta.

        ParÃ¡metros
        ----------
        name : str
            Nombre de la polÃ­tica (p. ej. "fijo", "decay", "var").
        """
        self.name = name

        # Series por batch
        self.regret_teorico = []
        self.regret_estimado = []
        self.regret_autoref = []

        # Series acumuladas
        self.cum_teorico = None
        self.cum_estimado = None
        self.cum_autoref = None

    # ---------------------------------------------------------
    # MÃTODO AUXILIAR: CÃLCULO DE REGRET INSTANTÃNEO
    # ---------------------------------------------------------
    def _calcula_regret(self, best_mean, means_vector, arm_to_idx, emparejados):
        """
        Calcula el regret instantÃ¡neo para un batch.

        ParÃ¡metros
        ----------
        best_mean : float
            Media del mejor brazo (real o estimada).
        means_vector : np.ndarray
            Vector de medias (reales o estimadas).
        arm_to_idx : dict
            Diccionario movieId ? Ã­ndice.
        emparejados : DataFrame
            PelÃ­culas realmente observadas en el batch.

        Retorna
        -------
        float
            Regret instantÃ¡neo del batch.
        """
        if emparejados is None or len(emparejados) == 0:
            return 0.0

        medias_batch = [
            means_vector[arm_to_idx[mid]]
            for mid in emparejados.movieId.values
        ]

        reward_batch = np.mean(medias_batch)
        return best_mean - reward_batch

    # ---------------------------------------------------------
    # MÃTODO PRINCIPAL: ACTUALIZACIÃN POR BATCH
    # ---------------------------------------------------------
    def update(self, best_mean_real, true_means, means_estimadas, arm_to_idx, emparejados):
            """
            Actualiza las tres mÃ©tricas de regret para un batch.
    
            ParÃ¡metros
            ----------
            best_mean_real : float
                Media real del mejor brazo.
            true_means : np.ndarray
                Vector de medias reales.
            means_estimadas : np.ndarray
                Vector de medias estimadas por el algoritmo.
            arm_to_idx : dict
                Diccionario movieId ? Ã­ndice.
            emparejados : DataFrame
                PelÃ­culas realmente observadas en el batch.
            """
    
            # 1. Regret teÃ³rico
            r_teorico = self._calcula_regret(best_mean_real, true_means, arm_to_idx, emparejados)
            self.regret_teorico.append(r_teorico)
    
            # 2. Regret estimado
            best_estimated_mean = means_estimadas.max()
            r_estimado = self._calcula_regret(best_estimated_mean, means_estimadas, arm_to_idx, emparejados)
            self.regret_estimado.append(r_estimado)
    
            # 3. Regret auto-referencial
            self.regret_autoref.append(r_estimado - r_teorico)
    
        
    
    # ---------------------------------------------------------
    # ACUMULADOS
    # ---------------------------------------------------------
    def compute_cumulative(self):
        """Calcula los regrets acumulados una vez finalizado el experimento."""
    
        # Si no hay datos, inicializa a listas vacÃ­as
        if len(self.regret_teorico) == 0:
            self.cum_teorico = []
            self.cum_estimado = []
            self.cum_autoref = []
            return
        
        self.cum_teorico  = np.cumsum(self.regret_teorico)
        self.cum_estimado = np.cumsum(self.regret_estimado)
        self.cum_autoref  = np.cumsum(self.regret_autoref)

        
