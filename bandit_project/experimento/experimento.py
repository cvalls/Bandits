import numpy as np
import pandas as pd
from bandit.bandit import Bandit
from bandit.runner import ExperimentRunner
from experimento.resultado_experimento import ResultadoExperimento


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
        #self.df["recompensa"] = pd.to_numeric(self.df["recompensa"], errors="coerce").astype("Int64")
        self.df["recompensa"] = pd.to_numeric(self.df["recompensa"], errors="coerce").astype(float)


    def ejecutar(self):
        bandit = Bandit(self.exp_pol, self.mediasReales, self.dctBrazos, self.claves_brazos)
        self.exp_pol.policy.inicializar(bandit.brazos)
        runner = ExperimentRunner(self.exp_pol.policy,
                                  self.args["general"]["batch_size"],
                                  self.args["general"]["slate_size"],
                                  self.args["general"]["modo"],
                                  self.args["general"]["bloque_debug"],
                                  self.args["general"]["lim_entropia_cat"])
    
        trazas = runner.run(self.df, bandit, self.index_replay)
    
        self.exp_pol.regret.compute_cumulative()
    
        resultado = ResultadoExperimento(
            exp_pol=self.exp_pol,
            bandit=bandit,
            trazas=trazas,
            cfg_general=self.args["general"]
        )
    
        # añadimos df al contexto diagnóstico
        resultado.contexto_diagnostico["df"] = self.df
    
        return resultado

