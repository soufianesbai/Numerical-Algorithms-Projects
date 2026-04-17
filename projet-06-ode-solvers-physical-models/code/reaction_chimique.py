from partie1 import (
    meth_n_step,
    meth_epsilon,
    step_rk4,
    step_euler,
    step_midpoint,
    step_heun,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def F(X, Y, Z, b):
    return np.array(
        [10 + b * Z - X - X * Y * Y, 200 * (X + X * Y * Y - Y), 50 * (Y - Z)],
        dtype=float,
    )


def plot_res_equations(bs):
    fig, axes = plt.subplots(1, len(bs), figsize=(4 * len(bs), 4), sharey=True)

    if len(bs) == 1:
        axes = [axes]  # pour garder une liste

    cmap = plt.get_cmap("viridis")  # tu peux aussi essayer 'tab10', 'plasma', etc.

    for i, b in enumerate(bs):
        X0, Y0, Z0 = 1.0, 0.0, 0.0
        y0 = np.array([X0, Y0, Z0], dtype=float)

        ts, ys = meth_n_step(
            y0,
            0.0,
            int(1e6),
            1e-6,
            lambda t, y: F(y[0], y[1], y[2], b),
            step_method=step_rk4,
        )

        Y = ys[:, 1]
        color = cmap(i / len(bs))
        ax = axes[i]
        ax.plot(ts, Y, color=color, label=f"b = {b:.3f}")
        ax.set_title(f"b = {b:.3f}")
        ax.set_xlabel("t")
        if i == 0:
            ax.set_ylabel("Y(t)")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Évolution de Y(t) pour différentes valeurs de b", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("resolution.png", bbox_inches="tight")


def res_equation(b):
    """
    Intègre le système en RK4 fixe et trace Y(t) sur la figure courante
    pour le paramètre b. Ne crée pas de nouvelle figure.
    """
    # Conditions initiales
    X0, Y0, Z0 = 1.0, 0.0, 0.0
    y0 = np.array([X0, Y0, Z0], dtype=float)

    # Intégration
    ts, ys = meth_n_step(
        y0,
        0.0,
        int(1e6),
        1e-6,
        lambda t, y: F(y[0], y[1], y[2], b),
        step_method=step_rk4,
    )

    # Extraction de Y
    Y = ys[:, 1]

    # Retour des données si besoin
    return Y


# Comment ne considérer que la phase périodique d’une solution ?
# Il suffit de trouver les pics de la courbe Y(t) et de ne garder que ceux-ci.
# Pour cela, on peut utiliser la fonction find_peaks de scipy.


def periode_solution(b):
    Y = res_equation(b)
    peaks, _ = find_peaks(Y)
    peak_times = Y[peaks]

    # 3. Calcul des intervalles
    periods = np.diff(peak_times)
    # print("Périodes mesurées :", periods)
    # print("Période moyenne ≃", periods.mean())
    return periods


def plot_bifurcation(save=True, filename="bifurcation.png"):
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import numpy as np

    bs = np.linspace(0.1, 0.2, 200)
    plt.figure(figsize=(10, 6))
    plt.xlabel("b")
    plt.ylabel("Y (pics)")
    plt.title("Diagramme de bifurcation")

    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.0
    y0 = np.array([X0, Y0, Z0], dtype=float)

    for b in bs:
        # Phase transitoire pour stabiliser la solution
        res = meth_n_step(
            y0,
            0.0,
            int(1e6),
            1e-6,
            lambda t, y: F(y[0], y[1], y[2], b),
            step_method=step_rk4,
        )
        y_start = res[1][-1]

        # Intégration d'observation (après transitoire)
        ts_obs, ys_obs = meth_n_step(
            y_start,
            0.0,
            int(1e6),
            1e-6,
            lambda t, y: F(y[0], y[1], y[2], b),
            step_method=step_rk4,
        )

        Y_obs = ys_obs[:, 1]

        # Détection des pics
        peaks, _ = find_peaks(Y_obs, prominence=0.1)
        Y_peaks = Y_obs[peaks]

        # Ajout au diagramme
        plt.scatter([b] * len(Y_peaks), Y_peaks, s=1, color="black")

    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    bs = np.linspace(0.1, 0.2, 5)
    # plot_res_equations(bs)
    plot_bifurcation()
    # on définit nos cibles
    # cibles = {2: None, 3: None, 4: None, 5: None, 'inf': None}
    # # balayage de b
    # bs = np.linspace(0.1, 0.2, 5)
    # for b in bs:
    #     per = periode_solution(b)
    #     if len(per)<2:
    #         continue
    #     # on regroupe les périodes approchées
    #     uniq = np.unique(np.round(per, 3))
    #     n = len(uniq)
    #     # repérer les périodes 2,3,4,5
    #     if n in cibles and cibles[n] is None:
    #         cibles[n] = (b, uniq)
    #     # période "∞" si beaucoup de classes
    #     if n>8 and cibles['inf'] is None:
    #         cibles['inf'] = (b, uniq)
    #     # si toutes trouvées, on sort
    #     if all(cibles[k] is not None for k in cibles):
    #         break

    # # on affiche et on sauve les courbes
    # for k, info in cibles.items():
    #     if info is None:
    #         print(f"Pas trouvé de solution de période {k}")
    #         continue
    #     b_trouvé, classes = info
    #     print(f"Période {k} → b = {b_trouvé:.3f} ; classes de périodes = {classes}")
    #     # on sauve le portrait temporel
    #     res_equation(b_trouvé)
