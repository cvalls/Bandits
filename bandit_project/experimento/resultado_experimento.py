# -*- coding: utf-8 -*-

# experimento/resultado_experimento.py

import numpy as np

class ResultadoExperimento:
    """
    Objeto de dominio que encapsula TODO el resultado de un experimento:
    - trazas del runner
    - bandit final
    - acumulados
    - contexto para diagnóstico
    - contexto para dashboard
    """

    def __init__(self, exp_pol, bandit, trazas, cfg_general):
        self.exp_pol = exp_pol
        self.nombre = exp_pol.nombre
        self.bandit = bandit
        self.trazas = trazas
        self.cfg = cfg_general

        # -----------------------------------------
        # 1. Extraer trazas crudas
        # -----------------------------------------
        self.rewards = trazas["rewards"]
        self.history_list = trazas["history_list"]
        self.historicoMeansPorBatch = trazas["historicoMeansPorBatch"]
        self.historicoRecomendados = trazas["historicoRecomendados"]
        self.ucb_mean_history = trazas["ucb_mean_history"]
        self.sz_num_emparejados_batch = trazas["sz_num_emparejados_batch"]

        # -----------------------------------------
        # 2. Acumulados
        # -----------------------------------------
        self.sz_rewards_acumulado = np.cumsum(self.rewards)
        self.sz_avg_reward_acum = (
            self.sz_rewards_acumulado / np.arange(1, len(self.rewards) + 1)
        )

        # -----------------------------------------
        # 3. Contexto para diagnóstico
        # -----------------------------------------
        self.contexto_diagnostico = {
            "df": None,  # se asigna en main
            "brazos": bandit.brazos,
            "historicoRecomendados": self.historicoRecomendados,
            "historicoMeansPorBatch": self.historicoMeansPorBatch,
            "sz_avg_reward_acum": self.sz_avg_reward_acum,
            "slate": cfg_general["slate_size"],
            "window_best": cfg_general["window_best"],
            "window_global": cfg_general["window_global"],
            "margen": cfg_general["margen"],
            "mean_slate_history": trazas["mean_slate_history"],
            "best_arm_effective_history": trazas["best_arm_effective_history"]
        }

        # -----------------------------------------
        # 4. Contexto para dashboard
        # -----------------------------------------
        self.contexto_dashboard = {
            "sz_num_emparejados_batch": self.sz_num_emparejados_batch,
            "history_list": self.history_list,
            "rewards": self.rewards,
            "sz_rewards_acumulado": self.sz_rewards_acumulado,
            "sz_avg_reward_acum": self.sz_avg_reward_acum,
            "ucb_mean_history": self.ucb_mean_history,
            "historicoMeansPorBatch": self.historicoMeansPorBatch,
            "window_variacion": cfg_general["window_variacion"],
        }
