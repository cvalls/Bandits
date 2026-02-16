# Bandits
Piloto de bandits sin contexto, recompenss bernouilli, gaussiana y categorica

# Bandits
Implementación completa de un sistema de *multi‑armed bandits* sin contexto, con soporte para recompensas **Bernoulli**, **Gaussianas** y **Categóricas**, incluyendo múltiples políticas, métricas avanzadas y un runner configurable para experimentación.

Este proyecto sirve como piloto para estudiar el comportamiento de distintas familias de bandits y comparar políticas clásicas como **UCB**, **Thompson Sampling** y **Boltzmann/Softmax** en entornos con distribuciones de recompensa diferentes.

---

## 🚀 Características principales

- **Tres tipos de bandits**:
  - **Bernoulli** (éxito/fracaso)
  - **Gaussiano** (recompensa continua)
  - **Categórico** (recompensa discreta multinomial)

- **Políticas implementadas**:
  - **UCB** (Upper Confidence Bound)
  - **Thompson Sampling**
  - **Boltzmann / Softmax**
  - Variantes específicas para cada tipo de bandit

- **Runner general** para ejecutar simulaciones:
  - Control de número de iteraciones
  - Criterios de parada (incluyendo entropía del *slate*)
  - Registro de métricas
  - Comparación entre políticas

- **Métricas avanzadas**:
  - Regret acumulado
  - Estabilidad del *slate*
  - Número efectivo de brazos (entropía)
  - Variación relativa de medias
  - Convergencia por batches
  - Tiempos de ejecución (policy, replay, total)

- **Soporte para slates** (selección de varios brazos por iteración)

