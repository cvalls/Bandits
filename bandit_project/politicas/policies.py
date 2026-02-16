# =========================================================
# POLÍTICAS DEL BANDIT (STRATEGY)
# =========================================================
#from typing import List
import numpy as np
import random
from scipy.stats import beta

class Policy:
    """
    Interfaz base para todas las políticas del bandit.
    Define los métodos comunes que el sistema espera.
    """

    def inicializar(self, brazos):
        """
        Inicializa la política cuando se conocen los brazos.
        Algunas políticas no necesitan inicialización.
        """
        pass

    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        """
        Actualiza parámetros internos de la política.
        UCB actualiza cotas, Boltzmann actualiza temperatura, etc.
        SOLO SE USA PARA ACTUALIZAR PARAMETROS GLOBALES.
        
        UCB lo usa para recalcular cotas UCB.
        Boltzmann lo usa para actualizar la temperatura.
        EpsilonGreedy lo usa para actualizar epsilon.
        """
        pass

    def seleccionar_slate(self, brazos, slate_size):
        """
        Selecciona un conjunto de brazos (slate).
        Debe ser implementado por cada política concreta.
        """
        raise NotImplementedError("seleccionar_slate debe implementarse en la subclase.")

    def update(self, idx, reward):
        """
        Actualización bayesiana o específica de la política.
        Solo lo usan BayesUCB, Thompson Sampling, Gaussian TS.
        Las demás políticas pueden ignorarlo.
        """
        pass

class UCBPolicy(Policy):
    
    # atributo de clase
    tipo_valido = "Ambos"
    
    def __init__(self, modo, alpha):
        self.modo = modo
        self.alpha = alpha
        
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        if self.modo == "Classic" :
            for b in brazos:
                b.actualizar_ucb_clasico(t_bandit, self.alpha)
        elif self.modo == "Tuned" :
            for b in brazos:
                b.actualizar_ucb_tuned(t_bandit)
        elif self.modo == "Varianza" :
            for b in brazos:
                b.actualizar_ucb_Varianza(t_bandit)
        elif self.modo == "KL" :
            for b in brazos:
                b.actualizar_ucb_KL(t_bandit)

    """
    Política UCB da igual el modo:
    - Ordena los brazos por su valor UCB (meansUCB o classicUCB).
    - Devuelve los primeros slate_size como recomendación.
    """
    def seleccionar_slate(self, brazos, slate_size):
        # ordenar por UCB descendente
        brazos_ordenados = sorted(brazos, key=lambda b: b.meansUCB, reverse=True)
        return brazos_ordenados[:slate_size]


class EpsilonGreedyPolicy(Policy):
    """
    Política epsilon-greedy:
    - Con probabilidad epsilon explora.
    - Con probabilidad 1 - epsilon explota.
    """
    tipo_valido = "Ambos"
    
    def __init__(self, modo, epsilon,
                 decay_rate=0.999, min_epsilon=0.01, max_epsilon=0.3,
                 factor_exp=1.0
    ):
        self.epsilon = epsilon
        self.modo = modo  # Fijo, Decay, Mixto, Varianza
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.factor_exp = factor_exp
        return


    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        # prueba 
        self.update_epsilon(brazos)
        #pass

    def seleccionar_slate(self, brazos, slate_size):
        num_brazos = len(brazos)

        # ------------------------------
        # Exploración
        # ------------------------------
        if np.random.rand() < self.epsilon:
            replace_flag = num_brazos < slate_size
            return list(np.random.choice(brazos, size=slate_size, replace=replace_flag))

        # ------------------------------
        # Explotación
        # ------------------------------
        brazos_ordenados = sorted(brazos, key=lambda b: b.effective_mean, reverse=True)
        recomendadas = brazos_ordenados[:slate_size]

        # ------------------------------
        # Relleno si faltan elementos
        # ------------------------------
        if len(recomendadas) < slate_size:
            faltan = slate_size - len(recomendadas)
            restantes = [b for b in brazos if b not in recomendadas]

            if len(restantes) >= faltan:
                extra = random.sample(restantes, faltan)  # sin reemplazo
            else:
                extra = random.choices(restantes, k=faltan)  # con reemplazo

            recomendadas = recomendadas + extra

        return recomendadas

    def update_epsilon(self, brazos):
        """
        Actualiza epsilon según el modo configurado.
        """
        eps_decay = 0
        eps_var = 0

        match self.modo:
            case "Fijo":
                return

            case "Decay":
                eps_decay = max(self.min_epsilon, self.epsilon * self.decay_rate)
                self.epsilon = eps_decay

            case "Varianza":
                varianzas = [b.classic_mean for b in brazos]
                std = np.std(varianzas)
                eps_var = np.clip(self.factor_exp * std, self.min_epsilon, self.max_epsilon)
                self.epsilon = eps_var

            case "Mixto":
                varianzas = [b.classic_mean for b in brazos]
                std = np.std(varianzas)
                eps_var = np.clip(self.factor_exp * std, self.min_epsilon, self.max_epsilon)
                eps_decay = max(self.min_epsilon, self.epsilon * self.decay_rate)
                self.epsilon = max(eps_decay, eps_var)

            case _:
                raise ValueError("Modo desconocido")

# =============================================================================
# Bayes UCB es un paradigma distinto a ucb, habla de distribuciones de probabilidad
# no de cotas. SOLO TIENE SENTIDO EN BERNOUILLI
# =============================================================================
class BayesUCBPolicy(Policy):
    
    tipo_valido = "Bernouilli"
    
    def __init__(self, c=3):
        self.c = c # control del cremiento del cuantil (entre 1 y 3)
        self.t = 1 # tiempo interno. no es el del bucle.

        # no se conocen al crear la politica todavia
        self.n_arms = None
        self.alpha = None # lista de exitos (1 por brazo)
        self.beta = None  # lista de fracasos (1 por brazo)

    def _ensure_init(self, brazos):
        # si no hay brazos, es que no se ha inicializado todavia. se hace ahora
        if self.n_arms is None:
            self.n_arms = len(brazos)
            self.alpha = np.ones(self.n_arms)
            self.beta = np.ones(self.n_arms)

    """
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        # Aquí garantizamos que alpha/beta estén inicializados
        # si ya lo estaban, esto retorna y pass
        # self._ensure_init(brazos)
        pass
    """
    def seleccionar_slate(self, brazos, slate_size):
        self._ensure_init(brazos)

        if self.t < 3: 
            q_level = 0.95
        else:
            q_level = 1 - 1 / (self.t * (np.log(self.t) ** self.c))

        # para cada brazo toma la distribucion posterior 
        # “¿Cuál es un valor alto pero razonable para la media de este brazo según mis creencias?”
        indices = [
            beta.ppf(q_level, self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]

        orden = np.argsort(indices)[::-1]
        slate = [brazos[i] for i in orden[:slate_size]]
        return slate

    def update(self, arm_idx, reward):
        # arm_idx es el índice del brazo en bandit.brazos
        # si hay exito se incrementa el alfa, si no el beta
        if reward == 1:
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx] += 1
        self.t += 1

# =============================================================================
# THOMSOM SAMPLING ES OTRO METODO. USA DISTRIBUCIONES DE PROBABILIDAD PRIOR POST COMO KTUCB
# Y MANEJA LA INCERTIDUMBRE. PERO SOLO VALE EN DISTRIBUCIONES BERNOUILLI
# =============================================================================
class ThomsonSamplingPolicy(Policy):
    tipo_valido = "Bernouilli"
    
    def __init__(self, modo, alpha_prior=1.0, beta_prior=1.0):
        
        self.modo = modo
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Diccionarios: brazo_id -> parámetros Beta
        self.alphas = {}
        self.betas = {}


    def _init_brazo_si_no_existe(self, brazo_id):

        if brazo_id not in self.alphas:
            self.alphas[brazo_id] = self.alpha_prior
            self.betas[brazo_id] = self.beta_prior


    def seleccionar_slate(self, brazos, slate_size):

        """
        Selecciona 'slate_size' brazos usando Thompson Sampling.
        Para cada brazo se muestrea theta ~ Beta(alpha, beta)
        y se eligen los brazos con mayor theta.
        """
        muestras = []
        
        for b in brazos:
            self._init_brazo_si_no_existe(b.idBrazo) # asegurar que el brazo se inicializa.
            alpha = self.alphas[b.idBrazo]
            beta = self.betas[b.idBrazo] # Muestreo TS
            
            # se toma un valor rnd de la distribucion NO SE PUEDE usar la media.
            theta = np.random.beta(alpha, beta)
            
            # guardamos el theta en el means para poder comparar luego.
            b.meansUCB = theta
            muestras.append((theta, b))

        # Ordenar por theta descendente y tomar los mejores
        muestras.sort(key=lambda x: x[0], reverse=True)
        slate = [b for (_, b) in muestras[:slate_size]]
        return slate
    
    def update(self, idx, reward):
        
        # Inicializar si es la primera vez
        self._init_brazo_si_no_existe(idx)
        
        # Actualizar posterior Beta
        if reward == 1:
            self.alphas[idx] += 1
        else:
            self.betas[idx] += 1

# =============================================================================
# BOLTZMANN ES OTRO METODO. USA DISTRIBUCIONES DE PROBABILIDAD PRIOR POST COMO KTUCB
#Y MANEJA LA INCERTIDUMBRE
# =============================================================================
class BoltzmannPolicy(Policy):
    tipo_valido = "Ambos"
    
    def __init__(self, modo, temperatura=0.5):
        
        self.modo = modo
        self.temp = temperatura
        self.temp0 = temperatura

    def seleccionar_slate(self, brazos, slate_size):
        # Extraer medias
        medias = np.array([b.effective_mean for b in brazos])
    
        # el exp explota si la media es grande. Con esto se evita
        medias_norm = medias - np.max(medias)
    
        # Clipping para evitar overflow si temp es muy pequeña
        scaled = np.clip(medias_norm / self.temp, -50, 50)
    
        # Softmax estable
        # esto da una array de exp (media(a) / temp)
        exp_vals = np.exp(scaled)
        
        # el denominador es la suma de los exp ee todas las medias
        probs = exp_vals / np.sum(exp_vals)
    
        # Guardar índice final en cada brazo (coherencia con UCB/TS)
        for b, p in zip(brazos, probs):
            b.meansUCB = p
    
        # Selección del slate mediante muestreo sin reemplazo
        # No se ordena el resultado se elige aleatoriamente ponderando por la probabilidad.
        idxs = np.random.choice(len(brazos), size=slate_size, replace=False, p=probs)
    
        slate = [brazos[i] for i in idxs]
        return slate

    def update_temperature(self, t):
        """
        Actualiza la temperatura según el modo configurado.
        """
        # el modo fijo es un softmax con la temperatura fija" 
        # el modo fijo es un softmax con la temperatura que disminuye con el tiempo"
        match self.modo:
            case "Fijo":
                return
            case "Adaptativo":
                t = max(t, 1)
                # academico simulated annealing (SA). garantiza maximo en este entorno.
                #self.temp = self.temp0 / np.log(t+2)

                # caso practico bandits. 
                self.temp = self.temp0 / np.sqrt(t)
                return
            case _:
                raise ValueError("Modo desconocido")
        return
                    
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):        
        self.update_temperature(t_softmax)           
        


# =============================================================================
# GAUSSIAN THOMSOM SAMPLING ES OTRO METODO. USA DISTRIBUCIONES DE PROBABILIDAD
# GAUSSIANAS NO BETAS
# =============================================================================
class GaussianThompsonSamplingPolicy(Policy):
    """
    Thompson Sampling Gaussiano (Normal-Normal conjugado).
    Compatible con recompensas reales (recompensa 1–5).
    """
    tipo_valido = "Continua"
    
    def __init__(self, mu0=0.0, tau0=1.0, sigma2=1.0):
        
        
        # Prior Normal(mu0, 1/tau0)
        self.mu0 = mu0
        self.tau0 = tau0

        # Varianza del ruido (likelihood)
        self.sigma2 = sigma2
        self.tau_noise = 1.0 / sigma2

        # Se inicializan en inicializar()
        self.mus = None
        self.taus = None

    def inicializar(self, brazos):
        """
        Inicializa los parámetros posteriores para cada brazo.
        """
        n = len(brazos)
        self.mus = np.full(n, self.mu0, dtype=float)
        self.taus = np.full(n, self.tau0, dtype=float)

    def update(self, idx, reward, reward_cat):
        """
        Actualiza la distribución posterior del brazo idx.
        Posterior Normal-Normal:
            tau_post = tau_prior + tau_noise
            mu_post  = (tau_prior*mu_prior + tau_noise*reward) / tau_post
        """
        tau_prior = self.taus[idx]
        mu_prior = self.mus[idx]

        tau_post = tau_prior + self.tau_noise
        mu_post = (tau_prior * mu_prior + self.tau_noise * reward) / tau_post

        self.taus[idx] = tau_post
        self.mus[idx] = mu_post


    def seleccionar_slate(self, brazos, slate_size):
        """
        Para cada brazo se muestrea:
            theta_i ~ Normal(mu_i, 1/tau_i)
        y se seleccionan los mejores slate_size.
        """
        # Muestreo de valores
        samples = np.random.normal(self.mus, 1.0 / np.sqrt(self.taus))

        # Guardar score en meansUCB (coherente con TS Beta y Boltzmann)
        for b, s in zip(brazos, samples):
            b.meansUCB = s

        # Seleccionar los mejores
        idxs = np.argsort(samples)[::-1][:slate_size]
        return [brazos[i] for i in idxs]



# =============================================================================
# GaussUCB es una politica determnista para recompensas continuas.
# No muestrea retorna los brazos ordenados por ma mediaUCB
# =============================================================================
class GaussianUCBPolicy(Policy):
    tipo_valido = "Continua"
    
    def __init__(self):
        pass
        
    # para Gauss UCB la media y la varianza quedan en cada brazo    
    # UCB es determinista luego se ordena y recupera el slate
    def seleccionar_slate(self, brazos, slate_size):
          # ordenar por UCB descendente
          brazos_ordenados = sorted(brazos, key=lambda b: b.meansUCB, reverse=True)
          return brazos_ordenados[:slate_size]
                      
    def actualizar_brazos(self, brazos, t_bandit, t_softmax):
        for b in brazos:
            b.actualizar_ucb_Gaussiano(t_bandit)


class BootstrappedThompsonPolicy(Policy):
    tipo_valido = "Ambos"

    def __init__(self, modo, sample_size = 100, percentile = 90):
        
        if ( sample_size < 1 ) :
            raise ValueError("BootstrappedThompsonPolicy: sample_size debe ser >= 1")
            
        if modo not in ("BTS", "BTS-P"):
            raise ValueError("BootstrappedThompsonPolicy: Modo desconocido ")
            
        if ( (modo == "BTS-P") and ( not ( 1 <= percentile <= 99 ) )) :
            raise ValueError("BootstrappedThompsonPolicy: Percentil invalido debe ir de 1 a 99")

        self.modo = modo
        self.sample_size = sample_size
        self.percentile = percentile
        
    def seleccionar_slate(self, brazos, slate_size):
    
        # para cada brazo, generar una media bootstrap
        bootstrap_means = []
        for b in brazos:
            if len(b.rewards_history) == 0:
                # si no hay datos, forzar exploración
                # se pone un numero grande pero random asi no hay empates (es es mas dificil)
                bootstrap_means.append(np.random.rand() * 1e6)
            else:
                # re-muestreo con reemplazo. No es necesario usar tooodo la lista
                # basta con unos cuantos
                # sample = np.random.choice(b.rewards_history, size=len(b.rewards_history), replace=True)
                sample = np.random.choice(b.rewards_history,
                                          size=min(self.sample_size, len(b.rewards_history)),
                                          replace=True)
                if self.modo == "BTS" :
                    valor = np.mean(sample)
                    
                elif self.modo == "BTS-P" :
                    valor = np.percentile(sample, self.percentile)
                    
                bootstrap_means.append(valor)
    
        # ordenar por media bootstrap
        indices = np.argsort(bootstrap_means)[::-1]
        return [brazos[i] for i in indices[:slate_size]]

