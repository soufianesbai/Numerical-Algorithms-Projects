import matplotlib.pyplot as plt
import numpy as np 
import bidiag 
import qr_algorithm
import compression_image

A = np.array(np.arange(0,25).reshape(5, 5),dtype='float64')

img_full = plt.imread("src/p3_takeoff_base.png")

# Extraction des matrices couleurs
img_r = img_full[:, :, 0] # composante rouge
img_g = img_full[:, :, 1] # composante verte
img_b = img_full[:, :, 2] # composante bleu

Q_left,BD,Q_right = bidiag.bidiagonalize_matrix(img_r)

# Les plots prennet un peu de temps a compiler puisque la matrice de la composante rouge de l'image est grande et sa bidiagonalisation prend du temps

def plot():
    """Plots the convergence and efficency of compression"""
    qr_algorithm.plot_convergence_svd(img_r,qr_algorithm.decomposition_svd,"Norme de l'écart entre S et sa diagonale avec np.linalg.qr") 

    qr_algorithm.plot_convergence_svd(BD,qr_algorithm.svd_opti,"Norme de l'écart entre S et sa diagonale avec qr_optimise")

    compression_image.efficacite_compression(img_r,[5,50,100,130,150,200])

#plot()