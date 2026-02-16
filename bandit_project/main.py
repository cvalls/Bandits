#import os
import numpy as np

# -----------------------------------------
# Módulos propios del proyecto
# -----------------------------------------
from experimento.experimento import Experimento
#from experimento.experimento_politica import ExperimentoPolitica

# Carga de datos
import data.crearconjuntodatos as sdata

# Bandit y componentes
#from bandit.policies import ClassicUCBPolicy, EpsilonGreedyPolicy
from bandit.utils import fijarMediasReales

from config.config_loader import ConfigLoader
from politicas.factory import create_experiment_policy

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
    MetricaMediaEfectivaSlate,
    MetricaGapEfectivo
)

from dashboard.graficos_comparativos import (
    plot_reward_acumulado,
    plot_regret_teorico_acumulado,
    plot_regret_estimado_acumulado,
    plot_avg_reward,
    plot_means_por_batch,
    plot_ucb_mean_history
)
from dashboard.graficos_effective import (
    plot_mean_slate_history,
    plot_best_arm_effective_history
)


from diagnostico.diagnostico import Diagnostico
from dashboard import BanditDashboard
from config.sweep import generar_configs_politica

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 1. Cargar configuración YAML
    # ---------------------------------------------------------
    cfg = ConfigLoader("config.yaml").load()

    dataset_cfg = cfg["dataset"]
    general_cfg = cfg["general"]
    policies_cfg = cfg["policies"]

    print("Batch:", general_cfg["batch_size"])
    print("Políticas definidas:", [p["name"] for p in policies_cfg])

    # se evita intentar llamar a politicas no aptas para el tipo de recompensa.
    #tipo_recompensa = dataset_cfg["tipo_de_recompensa"]
    #cfg.validarConfiguracion(tipo_recompensa, policies_cfg)
    
   


    # ---------------------------------------------------------
    # 2. Cargar dataset desde YAML
    # ---------------------------------------------------------
    df = sdata.cargarPeliculasElegidas( dataset_cfg )
  
    # ---------------------------------------------------------
    # 3. Preparar estructuras del experimento
    # ---------------------------------------------------------
    index_replay = df.groupby("movieId").indices
        
    
    claves_brazos = df.movieId.unique()
    dctBrazos = {mid: i for i, mid in enumerate(claves_brazos)}

    mediasReales = fijarMediasReales(df, claves_brazos)

    # ---------------------------------------------------------
    # 4. Crear políticas (de momento hardcodeadas)
    #    → En el siguiente paso haremos PolicyFactory
    # ---------------------------------------------------------
    
    # crear instancias reales de políticas
    #lista_ex_pol = [create_experiment_policy(pol_cfg) for pol_cfg in policies_cfg]
    
    # crear instancias reales de políticas (con sweep si aplica)
    lista_ex_pol = []
    
    for pol_cfg in policies_cfg:
        nombre_base = pol_cfg["name"]
        sweep_cfg = cfg.get("sweeps", {}).get(nombre_base, None)
    
        if sweep_cfg is None:
            # sin barrido: una sola instancia
            
            # en el caso de categorias, hay que saber cuantos brazos hay.
            # eso queda en la politica en vez de en los brazos.
            lista_ex_pol.append(create_experiment_policy(pol_cfg))
        else:
            # con barrido: múltiples combinaciones
            for cfg_variada, override in generar_configs_politica(pol_cfg, sweep_cfg):
                ex_pol = create_experiment_policy(cfg_variada)
                sufijo = "_".join(f"{k}={v}" for k, v in override.items())
                ex_pol.nombre = f"{nombre_base}__{sufijo}"
                lista_ex_pol.append(ex_pol)

    
    

    trazasexperimento = {}
    resultados ={}

    # ---------------------------------------------------------
    # 5. Ejecutar cada política
    # ---------------------------------------------------------
    # diagnóstico
    metricas = [
        MetricaMejorBrazoFinal(),
        MetricaDatosMejorPeli(),
        MetricaFrecuenciaMejorBrazo(window=500),
        MetricaEstabilidadSlate(slate=general_cfg["slate_size"]),
        MetricaEntropiaRecomendaciones(),
        MetricaVariacionRewards(window=500),
        MetricaPrimerBatchConvergencia(margen=5),
        MetricaVariacionRelativaMedias(window=500),
        MetricaMediaEfectivaSlate(),
        MetricaGapEfectivo()
    ]

    for exp_pol in lista_ex_pol:
        print(f"\n=== Inicio Ejecución de política: {exp_pol.policy.__class__.__name__} (modo={exp_pol.modo}) ===")
        experimento =  Experimento(cfg,exp_pol,df, index_replay,
                                  claves_brazos,dctBrazos,mediasReales)
        resultado = experimento.ejecutar()
     
        # Diagnóstico
        diag = Diagnostico(metricas)
        resultados_diag = diag.ejecutar(resultado.contexto_diagnostico)
        diag.imprimirMetricas(resultados_diag)
     
        # Dashboard
        panel = BanditDashboard(resultado)
        panel.render()
     
        # Guardar para comparativas
        resultados[exp_pol.nombre] = resultado
        print(f"\n=== Fin Ejecucion de política: {exp_pol.policy.__class__.__name__} (modo={exp_pol.modo}) ===")

      


    # ---------------------------------------------------------
    # 6. Gráficos comparativos
    # ---------------------------------------------------------

    print("\nGenerando gráficos comparativos...\n")
    plot_reward_acumulado(resultados)
    plot_regret_teorico_acumulado(resultados)
    plot_regret_estimado_acumulado(resultados)
    plot_avg_reward(resultados)
    plot_means_por_batch(resultados)
    plot_ucb_mean_history(resultados)
    plot_mean_slate_history(resultados)
    plot_best_arm_effective_history(resultados)