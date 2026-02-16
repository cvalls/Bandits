import time
import numpy as np
from .replay import filtrar_eventos_replay
from politicas import CategoricalBasePolicy

# =========================================================
# RUNNER GENÉRICO DEL EXPERIMENTO
# =========================================================
# hace solo: seleccionar brazos, hacer replay, actualizar, notificar métricas
# devuelve: rewards, historicoMeansPorBatch, historicoRecomendados, ucb_mean_history, sz_num_emparejados_batch

class ExperimentRunner:
    """
    Ejecuta un experimento de bandit con una política dada.
    Separa el bucle principal de la lógica del bandit.
    """

    def __init__(self, policy, batch_size, slate_size, modo, bloque, lim_entropia_cat):
        self.policy = policy
        self.batch_size = batch_size
        self.slate_size = slate_size
        self.modo = modo
        self.bloque = bloque
        self.limiteEntropia = lim_entropia_cat

    def run(self, df, bandit, index_replay):
        """
        Ejecuta el experimento completo y devuelve todas las trazas necesarias.
        """

        # ---------------------------------------------------------
        # Inicialización de contadores y trazas
        # ---------------------------------------------------------
        max_time = df.shape[0]

        rewards = []
        history_list = []
        historico_means_por_batch = []
        historico_recomendados = []
        ucb_mean_history = []
        num_emparejados_por_batch = []

        # buffers para graficar medias efectivas y probabilidad de premio
        mean_slate_history = []
        best_arm_effective_history = []


        # Contadores de tiempo
        time_policy = 0.0
        time_replay = 0.0
        time_update = 0.0
        time_total = 0.0

        t_bandit = 1
        t_softmax = 1

        # ---------------------------------------------------------
        # Bucle principal del experimento
        # ---------------------------------------------------------
        for t_dataset in range(0, max_time, self.batch_size):

            vuelta_start = time.perf_counter()

            # ------------------------------
            # 1. Selección de brazos
            # ------------------------------
            t0 = time.perf_counter()
            slate_recomendado = bandit.seleccionar(self.slate_size)
            
            
            # Media efectiva promedio del slate
            if slate_recomendado is not None and len(slate_recomendado) > 0:
                mean_slate = np.mean([b.effective_mean for b in slate_recomendado])
            else:
                mean_slate = np.nan
            mean_slate_history.append(mean_slate)
            
            # Media efectiva del mejor brazo (según estimación actual)
            best_effective = max(b.effective_mean for b in bandit.brazos)
            best_arm_effective_history.append(best_effective)

            
            
            # cada vez que se toma un brazo en el slate se ha usado y hay que
            # incrementar su contador. Haya o no haya premio.
            if slate_recomendado is not None:
                for brazo in slate_recomendado :
                    brazo.n_plays += 1
            time_policy += time.perf_counter() - t0

            # ------------------------------
            # 2. Replay
            # ------------------------------
            t1 = time.perf_counter()
            eventos_emparejados = filtrar_eventos_replay(
                index_replay, df, slate_recomendado, t_dataset, self.batch_size
            )
          
            
            time_replay += time.perf_counter() - t1

            # ------------------------------
            # 3. Actualización del bandit
            # ------------------------------
            t2 = time.perf_counter()
            bandit.actualizar(
                eventos_emparejados,
                t_dataset,
                t_bandit,
                t_softmax,
                history_list,
                num_emparejados_por_batch
            )
            time_update += time.perf_counter() - t2

            # el contador de tiempo depende de si hay emparejados.
            # si se incrementa cuando no hay ucb cree que ha explorado mas 
            # es decir el tiempo en ucb y boltzman depende de que haya habido
            # emparejamiento
            # para ucb y boltzmann t son cuántas observaciones reales tiene el
            # algoritmo para justificar reducir el bono de incertidumbre..
            if eventos_emparejados is not None:
                t_bandit += len(eventos_emparejados)
            
            # softmax usa el numero de vueltas para enfriar la temperatura,
            # haya o no emparejamiento
            t_softmax += 1;
            
            # ------------------------------
            # 4. Guardar trazas
            # ------------------------------
            historico_recomendados.append(slate_recomendado)
            historico_means_por_batch.append([b.classic_mean for b in bandit.brazos])
            ucb_mean_history.append(np.mean([b.ucb for b in bandit.brazos]))

            if eventos_emparejados is not None and len(eventos_emparejados) > 0:
                rewards.append(eventos_emparejados.recompensa.mean())
            else:
                rewards.append(np.nan)

            # ------------------------------
            # 5. Tiempo total de la vuelta
            # ------------------------------
            time_total += time.perf_counter() - vuelta_start

            # ------------------------------
            # 6. Logging periódico
            # ------------------------------...
            if t_dataset % 100000 == 0:
                print(f"\n=== Progreso: t = {t_dataset} (vueltas reales = {t_dataset // self.batch_size}) ===")
                print(f"  Tiempo acumulado policy: {time_policy:.2f}s")
                print(f"  Tiempo acumulado replay: {time_replay:.2f}s")
                print(f"  Tiempo acumulado update: {time_update:.2f}s")
                print(f"  Tiempo total acumulado: {time_total:.2f}s")
                print(f"  Eventos emparejados acumulados: {sum(num_emparejados_por_batch)}")



            # condicion de parada de la politica una vez se ha estabilizado.
            # solo para politicas categoricas.
            if isinstance(bandit.policy, CategoricalBasePolicy):
                H = bandit.policy.dameEntropiaMaxima()
                if ( H < self.limiteEntropia ):
                    print(f"Parada por entropia baja {H} limite: {self.limiteEntropia}")
                    break
            
            
            if ( self.modo == "DEBUG" ):
                if t_dataset > 0 and t_dataset % self.bloque == 0:
                    print(f" Modo DEBUG, bloque {self.bloque}")
                    break
            
        # ---------------------------------------------------------
        # Devolver trazas + bandit actualizado
        # ---------------------------------------------------------
        return {
             "rewards": rewards,
             "history_list": history_list,
             "historicoMeansPorBatch": historico_means_por_batch,
             "historicoRecomendados": historico_recomendados,
             "ucb_mean_history": ucb_mean_history,
             "sz_num_emparejados_batch": num_emparejados_por_batch,
             "mean_slate_history": mean_slate_history,
             "best_arm_effective_history": best_arm_effective_history,
             "bandit": bandit
        }


