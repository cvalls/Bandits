import numpy as np
import pandas as pd
#from bandit.brazos import Brazo as cs
#from bandit.brazos import crear_Brazo
from bandit.bandit import Bandit
from bandit.runner import ExperimentRunner
from diagnostico.diagnostico import Diagnostico
from dashboard import dashboard

# Métricas
from metricas import (
    MetricaMejorBrazoFinal,
    MetricaDatosMejorPeli,
    MetricaFrecuenciaMejorBrazo,
    MetricaEstabilidadSlate,
    MetricaEntropiaRecomendaciones,
    MetricaVariacionRewards,
    MetricaPrimerBatchConvergencia,
    MetricaVariacionRelativaMedias,
)




class Experimento:
    def __init__(self, args, exp_pol, df, index_replay, claves_brazos, dctBrazos, mediasReales):
        self.args = args
        self.exp_pol = exp_pol
        self.df = df
        self.index_replay = index_replay
        self.claves_brazos = claves_brazos
        self.dctBrazos = dctBrazos
        self.mediasReales = mediasReales
        
        # NORMALIZACIÓN DE TIPOS AQUÍ
        self.df["movieId"] = pd.to_numeric(self.df["movieId"], errors="coerce").astype("Int64")
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce").astype("float")

        
    def ejecutar(self):
        # crear brazos
        #brazos = [crear_Brazo(mid) for mid in self.claves_brazos]

        # crear bandit
        bandit = Bandit(
            #self.exp_pol.policy,
            self.exp_pol,
            self.mediasReales,
            self.dctBrazos,
            self.claves_brazos
        )

        # crear runner
        runner = ExperimentRunner(
            self.exp_pol.policy,
            self.args["general"]["batch_size"],
            self.args["general"]["slate_size"]
        )

        # ejecutar
        trazas = runner.run(self.df, bandit, self.index_replay)

        # procesar trazas
        rewards = trazas["rewards"]
        history_list = trazas["history_list"]
        historicoMeansPorBatch = trazas["historicoMeansPorBatch"]
        historicoRecomendados = trazas["historicoRecomendados"]
        ucb_mean_history = trazas["ucb_mean_history"]
        sz_num_emparejados_batch = trazas["sz_num_emparejados_batch"]

        # regret
        self.exp_pol.regret.compute_cumulative()

        # acumulados
        sz_rewards_acumulado = np.cumsum(rewards)
        sz_avg_reward_acum = np.cumsum(rewards) / np.linspace(1, len(rewards), len(rewards))

        
        
        # diagnóstico
        metricas = [
            MetricaMejorBrazoFinal(),
            MetricaDatosMejorPeli(),
            MetricaFrecuenciaMejorBrazo(window=500),
            MetricaEstabilidadSlate(slate=self.args["general"]["slate_size"]),
            MetricaEntropiaRecomendaciones(),
            MetricaVariacionRewards(window=500),
            MetricaPrimerBatchConvergencia(margen=5),
            MetricaVariacionRelativaMedias(window=500)
        ]

        diag = Diagnostico(metricas)
        resultados_diag = diag.ejecutar(
            df=self.df,
            bandit=bandit,
            historicoRecomendados=historicoRecomendados,
            historicoMeansPorBatch=historicoMeansPorBatch,
            sz_avg_reward_acum=sz_avg_reward_acum,
            slate=self.args["general"]["slate_size"],
            window_best=500,
            window_global=500,
            margen=20
        )
        diag.imprimirMetricas(resultados_diag)

        # dashboard individual
        panel = dashboard.BanditDashboard(
            sz_num_emparejados_batch,
            self.df,
            history_list,
            rewards,
            sz_rewards_acumulado,
            sz_avg_reward_acum,
            ucb_mean_history,
            historicoMeansPorBatch,
            window_variacion=500
        )
        panel.render()

        # devolver resultados
        return {
            "rewards": rewards,
            "sz_rewards_acumulado": sz_rewards_acumulado,
            "sz_avg_reward_acum": sz_avg_reward_acum,
            "historicoRecomendados": historicoRecomendados,
            "historicoMeansPorBatch": historicoMeansPorBatch,
            "ucb_mean_history": ucb_mean_history,
            "regret": self.exp_pol.regret,
            "mejorBrazoReal": bandit.mejorBrazoReal,
        }
