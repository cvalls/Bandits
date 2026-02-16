# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 19:05:39 2026

@author: Carlos
"""

import yaml

class ConfigLoader:
    """
    Carga y valida la configuración desde un archivo YAML.
    """

    def __init__(self, path_config):
        self.path_config = path_config
        self.config = None

    def load(self):
        """
        Carga el YAML y devuelve un diccionario con la configuración.
        """
        with open(self.path_config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._validar()
        return self.config

    def _validarConfiguracion (self, tipo_recompensa, policies_cfg) :
        from politicas.policies import (GaussianThompsonSamplingPolicy,
                                        ThomsonSamplingPolicy,
                                        UCBPolicy,
                                        EpsilonGreedyPolicy,
                                        BayesUCBPolicy,
                                        BoltzmannPolicy,
                                        GaussianUCBPolicy,
                                        BootstrappedThompsonPolicy)
        from politicas.Categorical_policies import (
                                        CategoricalThompsonSamplingPolicy,
                                        CategoricalUCBPolicy,
                                        CategoricalBoltzmannPolicy)

        # diccionario con todas las politicas a usar.
        policy_registry = {
            "UCBPolicy":UCBPolicy,
            "EpsilonGreedyPolicy":EpsilonGreedyPolicy,
            "BayesUCBPolicy":BayesUCBPolicy,
            "ThomsonSamplingPolicy":ThomsonSamplingPolicy,
            "BoltzmannPolicy":BoltzmannPolicy,
            "GaussianThompsonSamplingPolicy":GaussianThompsonSamplingPolicy,
            "GaussianUCBPolicy":GaussianUCBPolicy,
            "BootstrappedThompsonPolicy":BootstrappedThompsonPolicy,
            "CategoricalThompsonSamplingPolicy":CategoricalThompsonSamplingPolicy,
            "CategoricalUCBPolicy":CategoricalUCBPolicy,
            "CategoricalBoltzmannPolicy":CategoricalBoltzmannPolicy
            }

        # se comprueban si las politicas indicadas en la configuracion son compatibles con 
        # el tipo de recompensa declarado en la propia configuracion. 
        # cada politica tiene un campo tipo_recompensa 
        
        # validacion de politicas pedidas.
        for p in policies_cfg :
            nombre = p["name"]
            cls = policy_registry[nombre]
            tipo_valido = getattr(cls, "tipo_valido")
            if tipo_valido != "Ambos" and tipo_valido != tipo_recompensa:
                raise ValueError( f"La política {nombre} solo acepta recompensas de tipo {tipo_valido}, "
                                  f"pero el dataset es de tipo {tipo_recompensa}" )


    def _validar(self):
        try:
            if "general" not in self.config:
                raise ValueError("El YAML debe contener una sección 'general'.")
    
            if "policies" not in self.config:
                raise ValueError("El YAML debe contener una sección 'policies'.")
    
            if not isinstance(self.config["policies"], list):
                raise ValueError("'policies' debe ser una lista de políticas.")
    
            self._validarConfiguracion(
                self.dataset["tipo_de_recompensa"],
                self.policies
            )
    
        except Exception as e:
            print("\n❌ ERROR EN CONFIGURACIÓN DEL YAML ❌")
            print(str(e))
            raise   # ← RELANZA LA EXCEPCIÓN

    
        
    # Accesos cómodos
    @property
    def general(self):
        return self.config.get("general", {})

    @property
    def policies(self):
        return self.config.get("policies", []) # devuelve una lista de diccionarios.
         
    @property
    def dataset(self):
        return self.config.get("dataset", {})
