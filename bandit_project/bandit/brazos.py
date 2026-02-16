from dataclasses import dataclass, field
#import math
import numpy as np
from scipy.stats import dirichlet

# Cuando se define __init__ en una dataclass, se anula el constructor generado automáticamente
# Al definir un nuevo __init__, la dataclass ya no genera uno que acepte todos los campos asi que el constructor
# seria brazo(id)
# Pero la factoria llama con varios parametros.
# Para tener campos opcionales se usa from typing import Optional y se pone delante del tipo classicUCB: optional float
# y esos campos deben ir los ultimos de la dataclass. y declarar con none de valor
# solucion se quita el constructor __init__ con loq ue queda el constructor por defecto.
@dataclass
class Brazo:
    idBrazo: int
    n_rewards: int           # numero de recompensas del brazo
    n_plays: int             # numero de llamadas al brazo (con o sin premio)
    p_hat: float             # probabilidad de premio num tiradas/num tiradas premiads
    sum_rewards: float       # suma de recompensas. 
    
    classic_mean: float      # suma recompensas / numero llamadas  media clasica
                             # es decir la media del valor del premio cuando aparece
    effective_mean: float    # p_hat * classic_mean media efectiva
                             # lo que realmente importa para decidir qué brazo es mejor
    M2: float                # para algoritmo welford 
    ucb: float               # termino del bonus ucb
    meansUCB: float          # suma de measn + bono
    meansReal: float
    
    last_reward: float       # para Thomsom Sampling Bernouilli
    sum_rewards_sq:float     # cuadrado de la recompensa
    
    rewards_history : list[float] # lista de recompensas en el brazo para bootstrap
    effective_mean_history : list[float] 
    p_hat_history : list[float] 
  

    # self.classic_mean es la media clasica
    # self.ucb es el bono
    # self.meansUCB es la media mas el bono que es lo que se usa para seleccionar slate
    # usa media clasica porque modela la distribucion del premio no la recompensa esperada
    def actualizar_ucb_Gaussiano(self, t_bandit):
        
        # si t_bandit es 0 o 1 el log peta. Lo protegemos
        t_bandit  = max(t_bandit, 2)    
        
        # proteger counts
        if self.n_rewards == 0:
           self.ucb = float("inf")
           self.meansUCB = float("inf")
           return

        if self.n_rewards > 1:
            mean_sq = self.classic_mean ** 2
            avg_sq = self.sum_rewards_sq / self.n_rewards
            var = avg_sq - mean_sq
            
            # evitar negativos por redondeo
            var = np.clip(var, 1e-9, None)    
        else:
            var = 1e-9
            
        bonus = np.sqrt((2 * var * np.log(t_bandit)) / self.n_rewards)
        self.ucb = bonus
        self.meansUCB = self.classic_mean + self.ucb

    # cálculo del bono UCB clásico.
    def actualizar_ucb_clasico(self, t_bandit, alpha):
        if self.n_rewards == 0:
            self.ucb = float('inf')
        else:
            # renombre interno: t → t_bandit
            self.ucb = alpha * np.sqrt((2 * np.log(t_bandit)) / self.n_rewards)

        self.meansUCB = self.effective_mean + self.ucb

    def actualizar_ucb_tuned(self, t):
        if self.n_rewards == 0:
            self.ucb = float("inf")
            self.meansUCB = float("inf")
            return
    
        n = self.n_rewards
        mean = self.classic_mean # para la varianza empirica
        
        # Varianza empírica
        var_emp = (self.sum_rewards_sq / n) - (mean ** 2)
        var_emp = max(var_emp, 0)  # evitar negativos por redondeo
        
        epsilon = 1e-6
        var_emp = max(var_emp, epsilon)

    
        # si var_emp es pequeño, para n grande V es muy pequeño y el bono colapsa.
        # demasiado rapido asi que deja de explorar y se queda en un brazo suboptimo.

        # V_i(n)
        V = var_emp + np.sqrt(2 * np.log(t) / n)
    
        # min(1/4, V)
        V = min(0.25, V)
    
        # evitar V demasiado pequeño
        V = max(V, epsilon)

        # UCB-Tuned
        bonus = np.sqrt((np.log(t) / n) * V)
    
        self.ucb = bonus
        self.meansUCB = self.effective_mean + bonus

    def actualizar_ucb_Varianza(self, t):
        if self.n_rewards == 0:
            self.ucb = float("inf")
            self.meansUCB = float("inf")
            return
    
        n = self.n_rewards
        mean = self.classic_mean # para la varianza empirica
    
        # Varianza empírica
        var_emp = (self.sum_rewards_sq / n) - (mean ** 2)
        var_emp = max(var_emp, 0)  # evitar negativos por redondeo
    
        # evitar varianza cero (casos mean=0 o mean=1)
        # si mean = 0 o 1, el primer termino va a cero y domina el segundo termino.
        # muy bajo si n es grande y muy grande si n es pequeño
        epsilon = 1e-6
        var_emp = max(var_emp, epsilon)
   
        primer_termino = np.sqrt(2 * var_emp * np.log(t)/n)
        segundo_termino = 3 * np.log(t) /n
    
        # UCB-Varianza
        bonus = primer_termino + segundo_termino 
    
        self.ucb = bonus
        self.meansUCB = self.effective_mean + bonus        


    # En el artículo de Garivier & Cappé (2011), recomiendan valores entre 1 y 3. aunauq puede ser hasta 5
    # el 3 es el recomendado.
    # a mayor c se explora mas. 1 es mas agresivo y explora menos.
    # hay casos donde kl peta. p= 0 ,1 p=mid o cuando t=1 (log(log 1) es inf)
    def actualizar_ucb_KL(self, t, c=3):
        if self.n_rewards == 0:
            self.ucb = float("inf")
            self.meansUCB = float("inf")
            return
    
        n = self.n_rewards
        p = self.effective_mean
        epsilon = 1e-6
        
        # saturar p para evitar 0 y 1 exactos
        p = min(max(p, epsilon), 1 - epsilon)
        
    
        # Evitar log(log(t)) cuando t es pequeño
        # se usa 3 para que el termino log(log(t) sea positivo.)
        if t < 3:
            rhs = np.log(t) / n
        else:
            rhs = (np.log(t) + c * np.log(np.log(t))) / n
        
        # Hay que buscar un q tal que la divergencia KL sea menor o igual que este valor rhs
    
        # búsqueda binaria en q ∈ [p, 1]
        # se busca un valor q que este entre la media empirica y 1. se forma el rango
        low, high = p, 1.0
        
        # Se busca el punto medio del intervalo. haciendo prueba y error
        for _ in range(25):  # suficiente precisión
        
            mid = (low + high) / 2
            mid = min(mid, 1 - epsilon)  # evitar log(0)
        
            # KL divergencia Bernoulli con casos extremos. ya no hay p= 0 o 1
            # se evito arriba
            kl = p * np.log(p / mid) + (1 - p) * np.log((1 - p) / (1 - mid))
            
            
            # Si mid es grande, bajamos high si no subimos low,
            # El valor mid (candidato a q) está demasiado lejos de la media observada p
            # esta demasiado lejos de p para ser creible, asi que hay que bajar el limite.
            if kl > rhs:
                high = mid
            # el valor de mid puede ser el final, pero queremos seguir probando asi que el 
            else:
                low = mid
                
        # el resultado final para q es el menor low una vez terminada la busqueda
        q = low  # solución aproximada
    
        self.ucb = q - p
        self.meansUCB = q
    
# No es un método de Máquina sino una función un objeto máquina. el constructor es __init__
# no se puede tocar los campos de la clase porque se cambiaría la clase no hay instancia todavía.
# hay que pasar todos los campos porque si no se queja.
# defino una factoría para crear el agente.
def crear_brazo(mid, mediasReales, dctBrazos): 

    idx = dctBrazos[mid]
    return Brazo(
        idBrazo=mid,
        n_rewards=0,
        n_plays=0,
        p_hat=0,
        sum_rewards=0.0,
        effective_mean = 0.0,
        classic_mean = 0.0, 
        M2 = 0.0,
        last_reward = 0,
        sum_rewards_sq=0.0,
        ucb= float("inf"),
        meansUCB=0.0,
        meansReal=mediasReales[idx],
        rewards_history = [],
        effective_mean_history = [],
        p_hat_history = []
    )
