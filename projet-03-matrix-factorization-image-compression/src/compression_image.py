import numpy as np
import qr_algorithm
import bidiag
import matplotlib.pyplot as plt

def compression(A,rang): 
    """This function compresses the matrix given as argument to the indicated rank, rang
            A : The matrix to be compressed
            rang : the rank of compression 
    """
    U,S,V,Y = qr_algorithm.decomposition_svd(A) # Dans le cas ou A est bidiagonal on utilise svd optimise
    S,U = qr_algorithm.ordonner_matrice(S,U)
    k=S.shape[0]
    for i in range(rang+1,k):
        S[i,i] = 0
    A_compresse = np.dot(U,np.dot(S,V))
    return A_compresse

def compression_optimise(BD,rang): # BD doit etre une matrice bidiagonal pour utilise lalgo qr optimise
    """This function compresses the matrix given as argument to the indicated rank, rang
            A : A bidiagonal matrix to be compressed
            rang : the rank of compression 
        """
    U,S,V,Y = qr_algorithm.svd_opti(BD) 
    S,U = qr_algorithm.ordonner_matrice(S,U)
    k=S.shape[0]
    for i in range(rang+1,k):
        S[i,i] = 0
    A_compresse = np.dot(U,np.dot(S,V))
    return A_compresse

def efficacite_compression(A,rang_list) : 
    """ This function calculates the distance between the real image and the compressed images for different ranks in rang_list. Then a graph is drawn to show the efficacity of the compression
        A : the matrix which is going to be compressed
        rang_list : a list of different compression ranks """
    distances = []
    for rang in rang_list:
        A_compresse = compression(A,rang)
        distance = np.linalg.norm(A - A_compresse,2)
        distances.append(distance)
    plt.plot(rang_list, distances)
    plt.xlabel('rang (nombre de valeurs singulières utilisées)')
    plt.ylabel('Distance entre l\'image réelle et compressée')
    plt.title('Efficacité de la compression en fonction du rang')
    plt.grid(True)
    plt.show()

def compression_take_off_base(src,rang):

    img_full = plt.imread(src)
    
    # Extraction des matrices couleurs
    img_r = img_full[:, :, 0] # composante rouge
    img_g = img_full[:, :, 1] # composante verte
    img_b = img_full[:, :, 2] # composante bleu

    # Compression des matrices couleurs au rang r
    img_r_compresse = compression_optimise(img_r,rang)
    img_g_compresse = compression_optimise(img_g,rang)
    img_b_compresse = compression_optimise(img_b,rang)

    # Reconstruction d'image compresse 
    img_compressed = np.stack((img_r_compresse, img_g_compresse, img_b_compresse), axis=-1)
    
    # Afficher l'image compressée
    plt.imshow(img_full)
    plt.imshow(img_compressed)
    plt.title(f"Image compressée avec le rang {rang}")
    plt.show()
    print("img_r",img_r)
    print("img_g",img_g)
    print("img_b",img_b)
    print("img_r_compressed",img_r_compresse)
    print("img_g_compressed",img_g_compresse)
    print("img_b_compressed",img_b_compresse)


# Il faut compiler du dossier pere de l'image pour voir la compression

#compression_take_off_base("src/p3_takeoff_base.png",130)