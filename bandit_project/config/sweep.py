# -*- coding: utf-8 -*-
import itertools

def generar_configs_politica(base_cfg, sweep_cfg):
    keys = list(sweep_cfg.keys())
    values = [sweep_cfg[k] for k in keys]

    for combo in itertools.product(*values):
        override = dict(zip(keys, combo))
        cfg = base_cfg.copy()
        cfg.update(override)
        yield cfg, override  # override sirve para nombrar el experimento


