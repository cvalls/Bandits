# =========================================================
# POLÍTICAS DEL BANDIT (STRATEGY)
# =========================================================
#from typing import List
import numpy as np
from scipy.stats import dirichlet

from .policies import Policy

class CategoricalBasePolicy(Policy) :
    # atributo de clase
    tipo_valido = "Categorica"

    def __init__(self,  pesos, categorias) :
    
        # incialmente no se conocen lo brazos.
        self.num_brazos = None;    
        
        self.num_categorias = len(categorias)
        
        # vector de pesos de categorias
        self.pesos = pesos
    
        
        # las categorias vienen definidas en la columna recompensa del df
        # es un array de la A a la J en este caso.
        self.categorias = categorias
    
        self.entropia_max = None
        return
    
    def inicializar(self, brazos):
        self.num_brazos = len(brazos)
        
        # aunque los parametros de la dirchlet deberian estar en cada brazo, 
        # por compatibilidad con la arquitectura existente se almacenan en
        # la politica. Asi que cada fila es el vector de parametros de una maquina
        # que tiene num_categorias
        self.alpha = np.ones((self.num_brazos, self.num_categorias))
        return
        
    # solo vale para actualizar parametros globales de la politica.
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        pass
    
    
    def update(self, idx_brazo, reward, reward_cat):
        
        # Convertimos la categoria en un indice para saber que valor de alpha toca
        idx_categoria = self.categorias.index(reward_cat) 
        
        # se actualiza la dirichlet del brazo premiado
        self.alpha[idx_brazo, idx_categoria] += 1


    def _entropia_categorica(self, alpha):
        p = alpha / alpha.sum()
        return -np.sum(p * np.log(p + 1e-12))

    def dameEntropiaMaxima(self):
        return self.entropia_max
    
    def _calcular_entropia (self, idxs) :
        entropia_slate = [self._entropia_categorica(self.alpha[i])
                             for i in idxs]
        self.entropia_max = max(entropia_slate)
        
    
class CategoricalThompsonSamplingPolicy(CategoricalBasePolicy) :
    
    def seleccionar_slate(self, brazos, slate_size):

        # 1. Tomar una muestra theta de cada brazo
        #    y convertirla en un valor esperado usando los pesos
        valores = []
        for i, b in enumerate(brazos):
            theta = np.random.dirichlet(self.alpha[i])
            valor = np.dot(theta, self.pesos)
            valores.append(valor)
            

        # 2. Ordenar los índices por valor esperado (descendente)
        idxs = np.argsort(valores)[::-1][:slate_size]

        #fijar entropia maxima del slate
        self._calcular_entropia (idxs)
        
        # 3. Devolver los brazos seleccionados
        slate = [brazos[i] for i in idxs]
        return slate


class CategoricalUCBPolicy(CategoricalBasePolicy) :
    
    def __init__(self, pesos, categorias, factorExploracion=1.0):
        super().__init__(pesos, categorias)
        
        self.factorExploracion = factorExploracion
        self.t_bandit = None
        
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        # Guardamos t_bandit para usarlo en seleccionar_slate
        self.t_bandit = t_bandit
    
    def seleccionar_slate(self, brazos, slate_size):
        """
        Selección UCB para bandits categóricos.
        - Usa alpha para estimar la distribución categórica
        - Convierte esa distribución en un valor esperado usando los pesos
        - Añade un término de exploración UCB clásico
        - Devuelve objetos brazo
        """

        valores_ucb = []
        t = self.t_bandit #tiempo marcado por el bandit solo cuando el bandit detecta emparejados.

        for i, b in enumerate(brazos):

            # 1. Probabilidades categóricas estimadas
            alpha_i = self.alpha[i]
            p = alpha_i / alpha_i.sum()

            # 2. Valor esperado según los pesos
            mu_hat = np.dot(p, self.pesos)

            # 3. Término de exploración UCB
            if b.n_plays > 0:
                bonus = self.factorExploracion *np.sqrt( np.log(t) / b.n_plays)
            else:
                bonus = float("inf")  # forzar exploración inicial

            valores_ucb.append(mu_hat + bonus)

        # 4. Seleccionar los mejores brazos
        idxs = np.argsort(valores_ucb)[::-1][:slate_size]
        
        #fijar entropia maxima del slate
        self._calcular_entropia (idxs)
        
        return [brazos[i] for i in idxs]


class CategoricalBoltzmannPolicy(CategoricalBasePolicy) :
    
    def __init__(self, pesos, categorias, t_softmax=1.0):
        super().__init__(pesos, categorias)
        
        self.t_softmax = t_softmax
        
    
    def seleccionar_slate(self, brazos, slate_size):
        """
        Selección Boltxman para bandits categóricos.
        - Devuelve objetos brazo
        """
        # 1. Calcular valor esperado de cada brazo
        valores = []
        for i, b in enumerate(brazos):
            alpha_i = self.alpha[i]
            p = alpha_i / alpha_i.sum()
            mu_hat = np.dot(p, self.pesos)
            valores.append(mu_hat)

        valores = np.array(valores)

        # 2. Softmax: exp(valores / T)
        logits = valores / self.t_softmax
        exp_logits = np.exp(logits - np.max(logits))  # estabilidad numérica
        probs = exp_logits / exp_logits.sum()

        # 3. Elegir slate_size brazos según la distribución
        idxs = np.random.choice(len(brazos), size=slate_size, replace=False, p=probs)

        self._calcular_entropia (idxs)

        return [brazos[i] for i in idxs]
