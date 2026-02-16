# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 18:42:15 2026

@author: Carlos
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_reward_acumulado(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        plt.plot(res.sz_rewards_acumulado, label=nombre)
    plt.title("Reward acumulado por política")
    plt.xlabel("Iteración")
    plt.ylabel("Reward acumulado")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_regret_estimado_acumulado(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        plt.plot(res.exp_pol.regret.cum_estimado, label=nombre)
    plt.title("Regret estimado acumulado por política")
    plt.xlabel("Iteración")
    plt.ylabel("Regret estimado acumulado")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_regret_teorico_acumulado(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        plt.plot(res.exp_pol.regret.cum_teorico, label=nombre)
    plt.title("Regret teórico acumulado por política")
    plt.xlabel("Iteración")
    plt.ylabel("Regret teórico acumulado")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_avg_reward(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        plt.plot(res.sz_avg_reward_acum, label=nombre)
    plt.title("Reward medio acumulado por política")
    plt.xlabel("Iteración")
    plt.ylabel("Reward medio")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_means_por_batch(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        mean_por_batch = [np.mean(m) for m in res.historicoMeansPorBatch]
        plt.plot(mean_por_batch, label=nombre)
    plt.title("Evolución de medias por batch")
    plt.xlabel("Batch")
    plt.ylabel("Media global de los brazos")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_ucb_mean_history(resultados):
    plt.figure(figsize=(12, 6))
    for nombre, res in resultados.items():
        ucb_hist = res.ucb_mean_history
        if ucb_hist is not None:
            plt.plot(ucb_hist, label=nombre)
    plt.title("Evolución de UCB mean (solo políticas UCB)")
    plt.xlabel("Iteración")
    plt.ylabel("UCB mean")
    plt.legend()
    plt.grid(True)
    plt.show()
