# -*- coding: utf-8 -*-
import numpy as np
from bandit.brazos import crear_brazo

class Bandit:
    """
    Bandit genérico:
    - mantiene brazos
    - actualiza estadísticas básicas
    - delega el cálculo de UCB / epsilon / etc. en la política
    - actualiza regret
    """

    def __init__(self, ex_pol, mediasReales, dctBrazos, claves_brazos):
        self.policy = ex_pol.policy
        self.mediasReales = mediasReales
        self.mejorBrazoReal = mediasReales.max()
        self.dctBrazos = dctBrazos
        self.regret_tracker = ex_pol.regret

        # Crear brazos
        self.brazos = [crear_brazo(mid, mediasReales, dctBrazos) for mid in claves_brazos]

    # version anterior a separa probabilidad de premio con media de premio.
    def _actualiza_estadistica_clasicaTodojunto(self, idx, brazo, reward, reward_cat) :
        
        # cada vez que se entra aqui es porque hay un premio.
        brazo.n_rewards += 1
        
        # recompensa para el bootstrapped
        brazo.rewards_history.append(reward)
        
        # recompensas estadistica clasica
        brazo.sum_rewards += reward
        
        # para ts se guarda el resultado suelto no el acumulado.
        brazo.last_reward = reward
        
        brazo.sum_rewards_sq += reward * reward
        brazo.classic_mean = brazo.sum_rewards / brazo.n_rewards


        # >>> Actualizar la política Bayes-UCB <<<
        # en el caso de bayes lo que se hace es aumentar exito o
        # fracaso del brazo
        if hasattr(self.policy, "update"):
            self.policy.update(idx, reward, reward_cat )


    def _actualiza_estadistica_clasica(self, idx, brazo, reward, reward_cat):
        """
        Actualiza todas las estadísticas clásicas del brazo:
        - n_plays: nº de veces mostrado (se incrementa en el runner)
        - n_rewards: nº de veces que ha habido recompensa
        - sum_rewards: suma de recompensas
        - sum_rewards_sq: suma de cuadrados de recompensas
        - rewards_history: lista de recompensas (para BTS)
        - p_hat: prob. estimada de aparición
        - r_hat: valor medio del premio cuando aparece
        - means: media efectiva = p_hat * r_hat
        - last_reward: última recompensa (para TS)
        Además, llama a policy.update(idx, reward) si existe.
        """
    
        # 1. Si hay recompensa positiva, actualizar contadores de recompensa
        if reward is not None and reward > 0:
            # nº de recompensas observadas
            brazo.n_rewards += 1
    
            # suma de recompensas
            brazo.sum_rewards += reward
    
            # suma de cuadrados (para varianza clásica / UCB tuned, etc.)
            brazo.sum_rewards_sq += reward * reward
    
            # historial de recompensas (para Bootstrapped Thompson)
            brazo.rewards_history.append(reward)
    
            # última recompensa (para TS, si se usa)
            brazo.last_reward = reward
            
        # 2. Estimar p_hat = probabilidad de aparición
        if brazo.n_plays > 0:
            brazo.p_hat = brazo.n_rewards / brazo.n_plays
        else:
            brazo.p_hat = 0.0
    
        # 3. Estimar r_hat = valor medio del premio cuando aparece
        if brazo.n_rewards > 0:
            brazo.classic_mean = brazo.sum_rewards / brazo.n_rewards
        else:
            brazo.classic_mean = 0.0
    
        # 4. Media efectiva del brazo (lo que usan UCB, BTS, etc.)
        brazo.effective_mean = brazo.p_hat * brazo.classic_mean
    
        # 5. Notificar a políticas bayesianas (BayesUCB, TS, Gaussian TS, etc.)
        if hasattr(self.policy, "update"):
            self.policy.update(idx, reward, reward_cat)
        
        brazo.effective_mean_history.append(brazo.effective_mean)
        brazo.p_hat_history.append(brazo.p_hat)


    # No considera la separacion de probabilidades
    def _actualiza_estadistica_welford(self, idx, brazo, reward):
        """
        Actualiza media y varianza del brazo usando el algoritmo de Welford.
        Guarda:
          - brazo.classic_mean  (media)
          - brazo.M2     (acumulador)
          - brazo.var    (varianza empírica)
        """
    
        brazo.n_rewards += 1
        n = brazo.n_rewards
        
        # recompensa para el bootstrapped
        brazo.rewards_history.append(reward)
    
        # estadistica clasica
        # delta entre el nuevo valor y la media anterior
        delta = reward - brazo.lassic_means
    
        # actualizar media
        brazo.classic_mean += delta / n
    
        # delta2 entre el nuevo valor y la nueva media
        delta2 = reward - brazo.classic_mean
    
        # actualizar acumulador M2
        brazo.M2 += delta * delta2
    
        # varianza empírica (si n > 1)
        if n > 1:
            brazo.var = brazo.M2 / (n - 1)
        else:
            brazo.var = 0.0

        # 2. Estimar p_hat = probabilidad de aparición
        if brazo.n_plays > 0 :
            brazo.p_hat = brazo.n_rewards / brazo.n_plays
        else:
            brazo.p_hat = 0.0
               
        # 4. Media efectiva del brazo (lo que usan UCB, BTS, etc.)
        brazo.effective_mean = brazo.p_hat * brazo.classic_mean
            
        if hasattr(self.policy, "update"):
            self.policy.update(idx, reward, reward_cat)
            
        brazo.effective_mean_history.append(brazo.effective_mean)
        brazo.p_hat_history.append(brazo.p_hat)

    def actualizar_welford(self, reward, reward_cat):
        """
        Actualiza Welford SOLO con recompensas positivas.
        Esto estima r_i = valor medio del premio cuando aparece.
        """
    
        if reward is None or reward <= 0:
            return  # no actualizar nada
    
        self.welford_count += 1
    
        delta = reward - self.welford_mean
        self.welford_mean += delta / self.welford_count
        delta2 = reward - self.welford_mean
        self.welford_M2 += delta * delta2
    

    # ---------------------------------------------------------
    # SELECCIÓN del slate depende de la politica
    # ---------------------------------------------------------
    def seleccionar(self, slate_size):
        return self.policy.seleccionar_slate(self.brazos, slate_size)


    # ---------------------------------------------------------
    # ACTUALIZACIÓN
    # ---------------------------------------------------------
    def actualizar(self, emparejados, t_dataset, t_bandit, t_softmax,
                   history_list, sz_num_emparejados_batch):
        """
        Actualiza estadísticas básicas de los brazos.
        La política se encarga de actualizar UCB / epsilon / etc.
        """

        # Si no hay emparejamientos
        if emparejados is None or len(emparejados) == 0:
            sz_num_emparejados_batch.append(0)
        else:
            sz_num_emparejados_batch.append(len(emparejados))

            # Actualizar estadísticas por cada evento
            for _, row in emparejados.iterrows():
                movie_id = row["movieId"]
                reward = row["recompensa"]
                reward_cat = row["recompensa_cat"]

                # Guardar histórico
                history_list.append((movie_id, reward, t_dataset))

                # Índice del brazo
                idx = self.dctBrazos[movie_id]
                brazo = self.brazos[idx]
                
                # se actualizan las estadisticas del brazo.
                self._actualiza_estadistica_clasica(idx, brazo, reward, reward_cat)
    
        # -----------------------------------------------------
        # Delegar en la política la actualización de UCB, temperatura o epsilon
        # esto actualiza parametros de la politica no del brazo. 
        # -----------------------------------------------------
        self.policy.actualizar_brazos(self.brazos, t_bandit, t_softmax)

        # -----------------------------------------------------
        # Actualizar regret
        # -----------------------------------------------------
        means_array = np.array([b.classic_mean for b in self.brazos])

        self.regret_tracker.update(
            self.mejorBrazoReal,
            self.mediasReales,
            means_array,
            self.dctBrazos,
            emparejados
        )
