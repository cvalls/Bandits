# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 23:11:03 2026

@author: Carlos
"""

from experimento.experimento_politica import ExperimentoPolitica
#from bandit.policies import UCBPolicy, EpsilonGreedyPolicy 
import politicas as policies_module

def create_experiment_policy(cfg):
    # 1. Nombre de la clase de la política
    class_name = cfg["name"]

    # 2. Intentar obtener la clase real
    try:
        cls = getattr(policies_module, class_name)
    except KeyError:
        raise ValueError(f"Política desconocida: '{class_name}'. "
                         f"Asegúrate de que el YAML usa el nombre de clase correcto.")

    # 3. Quitar "name" para pasar solo parámetros válidos al constructor
    params = {k: v for k, v in cfg.items() if k != "name"}

    # 4. Intentar instanciar la política con los parámetros del YAML
    try:
        pol = cls(**params)
    except TypeError as e:
        raise ValueError(
            f"Error al crear la política '{class_name}' con parámetros {params}. "
            f"Revisa el YAML. Detalle: {e}"
        )

    # 5. Crear el wrapper ExperimentoPolitica
    nombre_exp = f"{class_name}_{params.get('modo', '')}".strip("_")
    return ExperimentoPolitica(nombre_exp, params.get("modo", None), pol)

