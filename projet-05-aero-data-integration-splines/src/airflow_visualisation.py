import numpy as np
import matplotlib.pyplot as plt
from spline import cubic_spline, eval_spline
from integration import integration_n_simpson
from load_foil import load_foil
import matplotlib.colors as colors

# Fonction pour calculer la pression et l'écoulement d'air autour de l'aile
def compute_airflow_pressure(ex, ey, ix, iy, num_lambda=10, num_points=100):
    # Déterminer les limites minimales et maximales pour les courbes
    h_min = np.min(iy)
    h_max = np.max(ey)

    x_min = np.min(ex)
    x_max = np.max(ex)
    x_values = np.linspace(x_min, x_max, num_points)  # Création des points x
    
    # Calcul des splines pour les courbes supérieure et inférieure de l'aile
    M_upper = cubic_spline(ex, ey)
    M_lower = cubic_spline(ix, iy)
    
    upper_curve = np.array([eval_spline(ex, ey, M_upper, x) for x in x_values])
    lower_curve = np.array([eval_spline(ix, iy, M_lower, x) for x in x_values])
    
    # Définition des lambdas et des listes pour les lignes de flux
    lambda_values = np.linspace(0, 1, num_lambda)
    flow_lines_upper = []
    flow_lines_lower = []
    lengths_upper = []
    lengths_lower = []
    
    # Calcul de la dérivée numérique pour les courbes d'écoulement
    def numerical_derivative(x_array, y_array, h=1e-6):
        return [(eval_spline(ex, ey, M_upper, x + h) - eval_spline(ex, ey, M_upper, x)) / h 
                for x in x_array]
        
    def numerical_derivative_lower(x_array, y_array, h=1e-6):
        return [(eval_spline(ix, iy, M_lower, x + h) - eval_spline(ix, iy, M_lower, x)) / h 
                for x in x_array]
    
    # Calcul des lignes de flux pour la partie supérieure de l'aile
    for lambda_val in lambda_values:
        y_flow = (1 - lambda_val) * upper_curve + lambda_val * 3 * h_max
        flow_lines_upper.append((x_values, y_flow))
        
        # Dérivée numérique pour la courbe supérieure
        if lambda_val == 0:
            deriv = numerical_derivative(x_values, y_flow)
        else:
            deriv_upper = numerical_derivative(x_values, upper_curve)
            deriv = [(1 - lambda_val) * d for d in deriv_upper]
        
        # Calcul de la longueur d'une ligne de flux à l'aide de l'intégration numérique
        def length_integrand(x_idx):
            x_idx_int = int(x_idx)
            if x_idx_int >= len(deriv) - 1:
                x_idx_int = len(deriv) - 2
            return np.sqrt(1 + deriv[x_idx_int]**2)
        
        length = integration_n_simpson(length_integrand, 0, len(x_values)-1, 100)
        lengths_upper.append(length)
    
    # Calcul des lignes de flux pour la partie inférieure de l'aile
    for lambda_val in lambda_values:
        y_flow = (1 - lambda_val) * lower_curve + lambda_val * 3 * h_min
        flow_lines_lower.append((x_values, y_flow))
        
        # Dérivée numérique pour la courbe inférieure
        if lambda_val == 0:
            deriv = numerical_derivative_lower(x_values, y_flow)
        else:
            deriv_lower = numerical_derivative_lower(x_values, lower_curve)
            deriv = [(1 - lambda_val) * d for d in deriv_lower]
        
        # Calcul de la longueur de la ligne de flux pour la courbe inférieure
        def length_integrand(x_idx):
            x_idx_int = int(x_idx)
            if x_idx_int >= len(deriv) - 1:
                x_idx_int = len(deriv) - 2
            return np.sqrt(1 + deriv[x_idx_int]**2)
        
        length = integration_n_simpson(length_integrand, 0, len(x_values)-1, 100)
        lengths_lower.append(length)
    
    # Calcul de la pression à partir des longueurs obtenues
    ref_length = lengths_upper[0]
    pressure_upper = [(ref_length / length)**2 for length in lengths_upper]
    pressure_lower = [(ref_length / length)**2 for length in lengths_lower]
    
    # Calcul de la différence de pression entre la partie inférieure et supérieure
    pressure_diff = [pl - pu for pl, pu in zip(pressure_lower, pressure_upper)]
    total_lift = sum(pressure_diff) / len(pressure_diff)
    
    return pressure_upper, pressure_lower, x_values, flow_lines_upper, flow_lines_lower, total_lift

# Fonction pour afficher l'écoulement et la pression
def plot_airflow_and_pressure(ex, ey, ix, iy, num_lambda=10, num_points=100):
    pressure_upper, pressure_lower, x_values, flow_lines_upper, flow_lines_lower, total_lift = \
        compute_airflow_pressure(ex, ey, ix, iy, num_lambda, num_points)
    
    # Création de la première figure pour l'affichage des lignes de flux
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    
    # Affichage de l'aile et des lignes de flux
    ax1.plot(ex, ey, 'k-', linewidth=2, label='Airfoil surface')
    ax1.plot(ix, iy, 'k-', linewidth=2)
    
    for i, (x, y) in enumerate(flow_lines_upper):
        if i == 0:
            ax1.plot(x, y, 'b-', alpha=0.5, label='Upper flow lines')
        else:
            ax1.plot(x, y, 'b-', alpha=0.5)
    
    for i, (x, y) in enumerate(flow_lines_lower):
        if i == 0:
            ax1.plot(x, y, 'r-', alpha=0.5, label='Lower flow lines')
        else:
            ax1.plot(x, y, 'r-', alpha=0.5)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Airflow around the Airfoil')
    ax1.grid(True)
    ax1.legend()
    
    ax1.text(0.05, 0.05, f'Estimated total lift: {total_lift:.4f}', 
            transform=ax1.transAxes, fontsize=10, va='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('rapport/img/airflow_visualisation.png')
    plt.close()
    
    # Création de la deuxième figure pour la carte de pression
    fig2 = plt.figure(figsize=(12, 8))  # Wider figure
    ax2 = fig2.add_subplot(111)
    
    # Définition des coordonnées pour la carte de pression
    h_min = np.min(iy) 
    h_max = np.max(ey)
    
    x_min = np.min(ex)
    x_max = np.max(ex)
    
    # Add more padding to x-axis
    x_padding = 0.3 * (x_max - x_min)
    y_padding = 0.1 * (3*h_max - 3*h_min)
    
    grid_size = 100
    x_grid = np.linspace(x_min - x_padding, x_max + x_padding, grid_size)
    y_grid = np.linspace(3*h_min - y_padding, 3*h_max + y_padding, grid_size)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calcul des intensités de l'écoulement sur la grille
    velocity = np.zeros_like(X)
    
    M_upper = cubic_spline(ex, ey)
    M_lower = cubic_spline(ix, iy)
    
    upper_curve = np.array([eval_spline(ex, ey, M_upper, x) for x in x_grid])
    lower_curve = np.array([eval_spline(ix, iy, M_lower, x) for x in x_grid])
    
    # Calcul des valeurs de vitesse à chaque point de la grille
    for i in range(grid_size):
        for j in range(grid_size):
            x_point = X[i, j]
            y_point = Y[i, j]
            
            if x_point < x_min - x_padding or x_point > x_max + x_padding:
                velocity[i, j] = 0
                continue
                
            x_idx = np.searchsorted(x_grid, x_point) - 1
            x_idx = np.clip(x_idx, 0, len(x_grid) - 2)
            
            if x_idx >= len(upper_curve) or x_idx < 0:
                velocity[i, j] = 0
                continue
                
            upper_y = upper_curve[x_idx]
            lower_y = lower_curve[x_idx]
            
            if y_point >= lower_y and y_point <= upper_y and x_point >= x_min and x_point <= x_max:
                velocity[i, j] = 0
            else:
                if y_point > upper_y:
                    lambda_y = (y_point - upper_y) / (3*h_max - upper_y)
                    lambda_y = min(1, max(0, lambda_y))
                    velocity[i, j] = 1 - lambda_y
                elif y_point < lower_y:
                    lambda_y = (lower_y - y_point) / (lower_y - 3*h_min)
                    lambda_y = min(1, max(0, lambda_y))
                    velocity[i, j] = 1 - lambda_y
                else:
                    velocity[i, j] = 0
                if x_point < x_min or x_point > x_max:
                    edge_dist = min(abs(x_point - x_min), abs(x_point - x_max))
                    dist_factor = min(1.0, edge_dist / x_padding)
                    velocity[i, j] *= 1.0 - dist_factor
    
    # Création de la carte de pression avec un colormap personnalisé
    colors_list = [(0, 0, 0), (0.33, 0, 0), (0.66, 0, 0), 
                   (1, 0, 0), (1, 0.33, 0), (1, 0.66, 0),
                   (1, 1, 0), (1, 1, 0.33), (1, 1, 0.66), (1, 1, 1)]
    
    cmap_name = 'flow_intensity'
    cm = colors.LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)
    
    # Update the extent to match the wider grid
    heatmap = ax2.imshow(velocity, extent=[x_min-x_padding, x_max+x_padding, 
                                          3*h_min-y_padding, 3*h_max+y_padding], 
                         origin='lower', cmap=cm, aspect='equal', 
                         vmin=0, vmax=1)
    
    # Affichage des contours de l'aile et de la carte de pression
    ax2.plot(ex, ey, 'k-', linewidth=1.5)
    ax2.plot(ix, iy, 'k-', linewidth=1.5)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Airflow Intensity Heatmap (Black = No Flow, White = High Flow)')
    
    cbar = fig2.colorbar(heatmap, ax=ax2)
    cbar.set_label('Relative Flow Intensity')
    
    plt.tight_layout()
    plt.savefig('rapport/img/pressure_map.png')
    plt.show()
    
    return total_lift

# Fonction principale pour charger l'aile et afficher les résultats
if __name__ == "__main__":
    dim, ex, ey, ix, iy = load_foil("test/m5.dat")

    lift = plot_airflow_and_pressure(ex, ey, ix, iy)
    print(f"The estimated lift for this airfoil is: {lift}")
