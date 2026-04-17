import numpy as np
import matplotlib.pyplot as plt
from partie1 import * 
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

def maltusien(t, N):
    return g * N

def velhulst(t,N):
    return g * N * (1 - N / kappa)

def lotka_volterra(t, K):
    N, P = K

    dN = N * (a - b * P)
    dP = P * (c * N - d)
    return np.array([dN, dP])


def test_malthusien(y0, graph=False, h=1e-4, n_steps=100, g=0.3, kappa=300):
    """
    Test du modèle de croissance malthusienne 
    
    y0: Condition initiale de la population
    graph: trace le graphique de la population
    return: Temps et populations
    """
    population = y0
    t0 = 0
    f = maltusien
    ts, ys = meth_n_step(population, t0, n_steps, h, f, step_rk4)

    ys = np.array([y.item() for y in ys])  
    if graph:
        plt.figure()
        plt.plot(ts, ys)
        plt.xlabel("Temps")
        plt.ylabel("Population")
        plt.title("Croissance malthusienne")
        plt.grid()
        plt.show()

        print("Dernière valeur de population :", ys[-1])
    return ts, ys


def test_verhulst(y0, graph=False, h=1e-4, n_steps=10000, g=0.3, kappa=300):
"""
    Test du modèle de croissance de Verhulst 
    
    y0: Condition initiale de la population
    graph: trace le graphique de la population
    g: Taux de croissance de la population (défaut: -0.3)
    kappa: Capacité de charge (défaut: 300)
    return: Temps et populations
    """
    population = y0
    t0 = 0
    f = velhulst
    ts, ys = meth_n_step(population, t0, n_steps, h, f, step_rk4)

    ys = np.array([y.item() for y in ys])  
    if graph:
        plt.figure()
        plt.plot(ts, ys)
        plt.xlabel("Temps")
        plt.ylabel("Population")
        plt.title("Croissance de Verhulst")
        plt.grid()
        plt.show()

        print("Dernière valeur de population :", ys[-1])
    return ts, ys


def test_lotka_volterra(y0, graph=False, h=1e-4, n_steps=1000):
    """
    Test du modèle Lotka-Volterra 
    
    y0: Condition initiale de la population (proies, prédateurs)
    graph:trace les graphiques de la population des proies et des prédateurs
    :return: Temps et populations des proies et des prédateurs
    """
    population = y0
    t0 = 0
    ts, ys = meth_n_step(population, t0, n_steps, h, lotka_volterra, step_rk4)

    ys = np.array(ys)  
    N = ys[:, 0]  # Proies
    P = ys[:, 1]  # Prédateurs
    if graph: 
        plt.figure(figsize=(10, 5))
        plt.plot(ts, N, label="Proies (N)")
        plt.plot(ts, P, label="Prédateurs (P)")
        plt.xlabel("Temps")
        plt.ylabel("Population")
        plt.title("Modèle de Lotka-Volterra")
        plt.legend()
        plt.grid()
        plt.show()

        # Courbe (N, P)
        plt.figure()
        plt.plot(N, P)
        plt.xlabel("Proies (N)")
        plt.ylabel("Prédateurs (P)")
        plt.title("Phase (N(t), P(t))")
        plt.grid()
        plt.show()
    return ts, ys


def period(ts,ys):
   """
    Calcule la période d'un modèle de Lotka-Volterra à partir des temps et des populations (proies).

    ts: Tableau de temps
    ys: Tableau des population (Proie/Prédateur)
    return: La période moyenne du système en fonction des pics de la population des proies
    """
    #Proie
    N = ys[:, 0]  
    peaks, _ = find_peaks(N)  
    peak_times = ts[peaks]
    intervals = np.diff(peak_times)
    return np.mean(intervals)


def plot_local_behaviour(y0, eps=0.1, num_points=10, h=1e-2, n_steps=3000):
    """
    Trace le comportement local des solutions à partir d'un point de départ donné y0 avec des conditions initiales proches.
    
    y0: Condition initiale de la population (proies, prédateurs)
    :return: None 
    """
    decalage = np.linspace(0, eps, num_points)
    points = []

    for d in decalage:
        points.append(y0 + d)

    plt.figure(figsize=(8, 6))
    for pt in points:
        ts, ys = meth_n_step(pt, 0, n_steps, h, lotka_volterra, step_rk4)
        ys = np.array(ys)
        plt.plot(ys[:, 0], ys[:, 1], alpha=0.7)

    plt.xlabel("Proies (N)")
    plt.ylabel("Prédateurs (P)")
    plt.title("Comportement local autour de y0 ")
    plt.grid()
    plt.show()
if __name__ == "__main__":
    a = 1 
    b = 0.4 
    c = 0.1
    d = 0.4 
    #y0 = np.array([3, 2])
    y0 = np.array([1,1])
    #ts,ys = test_lotka_volterra(y0,True)
    #print(period(ts,ys))
    plot_local_behaviour(y0)
