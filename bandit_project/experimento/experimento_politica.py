import os
import numpy as np
import pandas as pd
from evaluacion.tracker import RegretTracker

class ExperimentoPolitica:
    def __init__(self, nombre, modo, policy):
        self.nombre = nombre
        self.modo = modo
        self.policy = policy
        self.regret = RegretTracker(nombre)
        self.runner = None  # se asigna luego