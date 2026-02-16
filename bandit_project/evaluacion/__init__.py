"""
MÃ³dulo de utilidades para el cÃ¡lculo, anÃ¡lisis y visualizaciÃ³n del regret
en algoritmos de Multi-Armed Bandits.

Incluye:
- RegretTracker: cÃ¡lculo de regret teÃ³rico, estimado y auto-referencial.
- Funciones auxiliares para mÃ©tricas.
- Exportadores a CSV/JSON.
- Herramientas de visualizaciÃ³n.

Este paquete estÃ¡ diseÃ±ado para ser independiente del algoritmo de bandits.
Cualquier polÃ­tica (e-greedy, UCB, Thompson, Softmax, EXP3, etc.) puede
utilizarlo sin modificaciones.
"""

from .tracker import RegretTracker
from .plotter import RegretPlotter
from .metrics import moving_average
from .exporter import RegretExporter

__all__ = ["RegretTracker", "plot_regret", "compute_metrics", "export_results"]

