import numpy as np
import matplotlib.pyplot as plt
import math
from partie_2 import ln_CORDIC, exp_CORDIC, arctan_CORDIC, tan_CORDIC

def graphiques_log2_approximatif():
    p_values = [4, 6, 8, 10]
    for p in p_values:
        log2_reel = math.log(2)
        log2_approx = 0
        terme = 0
        n = 1
        approximations = []
        erreurs = []

        while True:
            terme = ((-1)**(n + 1)) / n
            log2_approx += terme  
            err = abs((log2_approx - log2_reel) / log2_approx)
            
            approximations.append(log2_approx)
            erreurs.append(err)
            
            if err < 2**(-p):
                break
            n += 1
        
        plt.figure(figsize=(12, 6))
        plt.plot(approximations, label=f"Fonction log(2) approximatif pour p={p}")
        plt.axhline(y=log2_reel, color='r', linestyle='--', label="log(2) réel")
        
        plt.fill_between(range(len(approximations)),
                         log2_reel - 2**(-p), log2_reel + 2**(-p),
                         color='green', alpha=0.2, label=f"Intervalle ±2^(-{p})")
        
        plt.title(f"Convergence de l'approximation de log(2) pour p={p}")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("log(2) approximatif")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(erreurs, label=f"Fonctionrreur pour p={p}")
        
        plt.axhline(y=2**(-p), color='g', linestyle='--', label=f"Limite 2^(-{p})")

        plt.title(f"Évolution de l'erreur relative pour p={p}")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Erreur relative")
        plt.legend()
        plt.grid(True)
        plt.show()

def graphiques_ln_CORDIC():
    x_values = np.logspace(0.01, 5, num=1000)
    approximations = []
    errors = []

    for x in x_values:
        ln_approx = ln_CORDIC(x)
        ln_real = math.log(x)
        error = abs(ln_approx - ln_real)
        
        approximations.append(ln_approx)
        errors.append(error)
        
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, approximations, label="ln_CORDIC", marker='o')
    plt.plot(x_values, [math.log(x) for x in x_values], label="ln(x) réel", linestyle='--')
    plt.title("Comparaison de ln_CORDIC et ln(x) réel")
    plt.xlabel("x")
    plt.ylabel("ln(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, errors, label="Erreur", marker='o', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Erreur entre ln_CORDIC et ln(x) réel")
    plt.xlabel("x")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.show()

def graphiques_exp_CORDIC():
    x_values = np.linspace(0, 5, 1000)
    approximations = []
    errors = []

    for x in x_values:
        exp_approx = exp_CORDIC(x)
        exp_real = math.exp(x)
        error = abs(exp_approx - exp_real)
        
        approximations.append(exp_approx)
        errors.append(error)
        
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, approximations, label="exp_CORDIC", marker='o')
    plt.plot(x_values, [math.exp(x) for x in x_values], label="exp(x) réel", linestyle='--')
    plt.title("Comparaison de exp_CORDIC et exp(x) réel")
    plt.xlabel("x")
    plt.ylabel("exp(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, errors, label="Erreur", marker='o', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Erreur entre exp_CORDIC et exp(x) réel")
    plt.xlabel("x")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.show()

def graphiques_arctan_CORDIC():
    x_values = np.linspace(-5, 5, 1000)
    approximations = []
    errors = []

    for x in x_values:
        arctan_approx = arctan_CORDIC(x)
        arctan_real = math.atan(x)
        error = abs(arctan_approx - arctan_real)

        approximations.append(arctan_approx)
        errors.append(error)

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, approximations, label="arctan_CORDIC", marker='o')
    plt.plot(x_values, [math.atan(x) for x in x_values], label="arctan(x) réel", linestyle='--')
    plt.title("Comparaison de arctan_CORDIC et arctan(x) réel")
    plt.xlabel("x")
    plt.ylabel("arctan(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, errors, label="Erreur", marker='o', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Erreur entre arctan_CORDIC et arctan(x) réel")
    plt.xlabel("x")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.show()

def graphiques_tan_CORDIC():
    x_values = np.linspace(-2*math.pi, 2*math.pi, 1000)
    approximations = []
    errors = []

    for x in x_values:
        tan_approx = tan_CORDIC(x)
        tan_real = math.tan(x)
        error = abs(tan_approx - tan_real)

        approximations.append(tan_approx)
        errors.append(error)

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, approximations, label="tan_CORDIC", marker='o')
    plt.plot(x_values, [math.tan(x) for x in x_values], label="tan(x) réel", linestyle='--')
    plt.title("Comparaison de tan_CORDIC et tan(x) réel")
    plt.xlabel("x")
    plt.ylabel("tan(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, errors, label="Erreur", marker='o', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title("Erreur entre tan_CORDIC et tan(x) réel")
    plt.xlabel("x")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid(True)
    plt.show()

graphiques_log2_approximatif()
graphiques_ln_CORDIC()
graphiques_exp_CORDIC()
graphiques_arctan_CORDIC()
graphiques_tan_CORDIC()

