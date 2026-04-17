import numpy as np
import bidiag 
import matplotlib.pyplot as plt
import householder 
import bidiag

def decomposition_svd(A,Nmax=100) : 
    """ This function performs singular salue decomposition (SVD) using bidiagonalization and QR decomposition """
    Q_left,BD,Q_right = bidiag.bidiagonalize_matrix(A)
    U = Q_left.copy()
    V = Q_right.copy()
    S = BD.copy()
    Y=[]
    #while np.linalg.norm(np.diag(S,1))/(np.linalg.norm(np.diag(S)) + np.linalg.norm(np.diag(S,1))) > 1e-10 :
    for i in range(Nmax) :
        S_diag = np.zeros_like(S)
        min_dim = min(S.shape)  # Find the minimum dimension (square portion)
        np.fill_diagonal(S_diag[:min_dim, :min_dim], np.diagonal(S[:min_dim, :min_dim]))  
        Y.append(np.linalg.norm(S-S_diag))
        Q1,R1 = np.linalg.qr(S.T)
        Q2,R2 = np.linalg.qr(R1.T)
        S = R2 
        U = np.dot(U,Q2)
        V = np.dot(Q1.T,V)
        # if(np.linalg.norm(S-S_diag) < 1e-10):
        #     break
        assert np.allclose(U @ S @ V, A), \
            f"Invariant violated at step {i}"
    return U,S,V,Y

def qr_opti(BD):
    """ This function performs an optimized QR decomposition of a bidiagonal matrix BD"""
    m,n=BD.shape
    Q=np.eye(m)
    R=BD.copy()
    for i in range(n):
        if (i<m-1):
            X=np.array([R[i,i],R[i+1,i]])
            alpha=np.linalg.norm(X)
            Y=np.array([alpha,0])
            U=householder.projection(X,Y)
            H=householder.householder_matrix(U)
            R[i:i+2, i] = Y
            if (i<n-1):
                R[i:i+2, i+1] = np.dot(H,R[i:i+2, i+1])
            Q[:, i:i+2] = np.dot(Q[:, i:i+2], H)
    return Q, R

def svd_opti(A,Nmax=100) : 
    """ This function performs singular salue decomposition (SVD) on a matrix A bidiagonal and QR decomposition optimized """
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    S = A.copy()
    Y=[]
    #while np.linalg.norm(np.diag(S,1))/(np.linalg.norm(np.diag(S)) + np.linalg.norm(np.diag(S,1))) > 1e-10 :
    for i in range(Nmax):
        S_diag = np.zeros_like(S)
        min_dim = min(S.shape)  # Find the minimum dimension (square portion)
        np.fill_diagonal(S_diag[:min_dim, :min_dim], np.diagonal(S[:min_dim, :min_dim])) 
        Y.append(np.linalg.norm(S-S_diag))
        Q1,R1 = qr_opti(S.T)
        Q2,R2 = qr_opti(R1.T)
        S = R2 
        U = np.dot(U,Q2)
        V = np.dot(Q1.T,V)
    return U,S,V,Y

def ordonner_matrice(S,U):
    """This function ensures the singular values in matrix S are non-negative and ordered in descending order, 
        while adjusting the corresponding columns in matrix U to maintain consistency in the SVD"""
    m, n = S.shape
    for i in range(min(m, n)):
        if S[i, i] < 0:
            S[i, i] = -S[i, i]
            U[:, i] = -U[:, i]

    for i in range(min(m, n) - 1):
        for j in range(i + 1, min(m, n)):
            if S[i][i] < S[j][j]:
                S[i][i], S[j][j] = S[j][j], S[i][i] # Échanger les éléments de S
                # Échanger les colonnes de U
                for k in range(m):
                    U[k][i], U[k][j] = U[k][j], U[k][i]

    return S,U

def plot_convergence_svd(A,methode,y_label) : 
    """This function applies the SVD decomposition methode given on matrix A 
        and plots the convergence of the matrix S towards a diagonal matrix over the iterations"""
    U,S,V,Y = methode(A)
    plt.plot(Y,linestyle='-')
    plt.xlabel("Itérations")
    plt.ylabel(y_label)  
    plt.grid()
    plt.show()
