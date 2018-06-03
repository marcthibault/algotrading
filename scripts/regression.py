import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.correlation_tools import cov_nearest
from itertools import permutations, combinations

class Regressor():

    def __init__(self):
        """
        Sizes : 
        P : Number of stocks
        T : Number of time steps
        D : Number of covariates
        J : function space size
        K : Number of factors

        Parameters :
        function_description : list, human readable description of used functions
        function_space : list, python functions
        Y : numpy array, Stock returns, size P x T
        X : numpy array, Covariates, size P x D
        Phi : numpy array, Functions of the covariates, size P x JD
        B : numpy array, size JD x K
        F : numpy array, Factors, size T x K
        G : numpy array, Function of the covariates matrix, size P x K
        Lambda : numpy array, loadings, size P x K
        Gamma : numpy array. Residuals from the parameters fit, size P x K
        eps : numpy array, Residuals from the global fit, size P x T
        """


        self.function = []

        self.Y = None
        self.X = None
        self.Phi = None
        self.B = None
        self.F = None
        self.G = None
        self.Lambda = None
        self.Gamma = None
        self.eps = None

        self.K = 1

    def fit_PPCA(self, Y, X, K, find_K = False):
        """
        Y : numpy array, returns
        X : numpy array, covariates
        K : int, number of factors
        find_K : bool. If True, evaluate the best K
        """

        P, T = Y.shape
        _, D = X.shape
        J = len(self.function)

        self.X = X
        self.Y = Y

        assert J > 0, "You must fill the function space before fitting"
        assert P == X.shape[0], "The size of Y and X must be the same on axis 0 (number of stocks)"

        self.Phi = np.zeros((P, J*D))
        
        ## building of the projection space
        current_index = 0
        for col in range(D):
            for f in self.function:
                self.Phi[:, current_index] = f(self.X[:, col])
                current_index += 1

        ## Computation of the estimation
        Inv_covariance = np.linalg.pinv(np.matmul(self.Phi.T, self.Phi))
        
        Projection = np.matmul(np.matmul(self.Phi, Inv_covariance),
                               self.Phi.T)

        Covariance = np.matmul(self.Y.T, np.matmul(Projection, self.Y))

        values, vectors = np.linalg.eigh(Covariance)
        
        if find_K:
            values_sorted = np.array(sorted(values, reverse=True))
            ratio = values_sorted[:-1] / values_sorted[1:]
            self.K = np.argsort(-ratio)[0] + 1
        else:
            self.K = K
        
        self.F = np.sqrt(T) * vectors[:, np.argsort(-values)[:self.K]]
    
        YF = np.matmul(self.Y, self.F)

        self.B = 1 / T * np.matmul(np.matmul(Inv_covariance, self.Phi.T),
                                   YF)
        self.G = 1 / T * np.matmul(Projection, YF)

        self.Lambda = 1 / T * YF
        self.Gamma = self.Lambda - self.G

        self.eps = self.Y - np.matmul(self.Lambda, self.F.T)

    def fit_PC(self, Y, K, find_K = False):
        """
        Y : numpy array, returns
        X : numpy array, covariates
        K : int, number of factors
        find_K : bool. If True, evaluate the best K
        """

        P, T = Y.shape
        J = len(self.function)

        self.Y = Y

        assert J > 0, "You must fill the function space before fitting"

        ## Computation of the estimation

        Covariance = np.matmul(self.Y.T, self.Y)

        values, vectors = np.linalg.eigh(Covariance)
        
        if find_K:
            values_sorted = np.array(sorted(values, reverse=True))
            ratio = values_sorted[:-1] / values_sorted[1:]
            self.K = np.argsort(-ratio)[0] + 1
        else:
            self.K = K
        
        self.F = np.sqrt(T) * vectors[:, np.argsort(-values)[:K]]
    
        YF = np.matmul(self.Y, self.F)

        self.G = 1 / T * YF

        self.eps = self.Y - np.matmul(self.G, self.F.T)

    def add_indicator(self, start, stop):
        """
        Add indicator between start and stop
        return 1 if x > start and x < stop
        """
        self.function.append(lambda x : (x > start) * (x < stop) * 1.0)


    def add_cube(self, li_parameters):
        """
        Add polynomial of degree 3 function
        """
        a, b, c, d = li_parameters
        self.function.append(lambda x : a * x ** 3 + b * x ** 2 + c * x + d)

    def add_square(self, li_parameters):
        """
        Add polynomial of degree 2 function
        """
        a, b, c = li_parameters
        self.function.append(lambda x : a * x**2 + b * x + c)

    def add_line(self, li_parameters):
        """
        Add polynomial of degree 1 function
        """
        a, b = li_parameters
        self.function.append(lambda x : a * x + b)

    def add_max_square(self, knot):
        knot = knot
        self.function.append(lambda x : (x > knot) * (x - knot) ** 2)

    def add_max_cube(self, knot):
        knot = knot
        self.function.append(lambda x : (x > knot) * (x - knot) ** 3)

    def add_square_splines(self, lower, higher, dimension):
        """
        Generate a square spline basis
        """
        self.add_line([0.0, 1.0])
        self.add_line([1.0, 0.0])
        self.add_square([1.0, 0.0, 0.0])
        for i in range(dimension - 3):
            knot = lower + (higher - lower) * (i+1) / (dimension - 2)
            self.add_max_square(knot)

    def add_cubic_splines(self, lower, higher, dimension):
        """
        Generate a square spline basis
        """
        self.add_line([0.0, 1.0])
        self.add_line([1.0, 0.0])
        self.add_square([1.0, 0.0, 0.0])
        self.add_cube([1.0, 0.0, 0.0, 0.0])
        for i in range(dimension - 4):
            knot = lower + (higher - lower) * (i+1) / (dimension - 3)
            self.add_max_cube(knot)

def co_diagonalize(A, B):
    """
    Return H such that :
    H A H.T = I
    H B H.T = D, with D diagonal

    Procedure described here :
    """
    Cholesky = np.linalg.cholesky(B)
    Cholesky_inv = np.linalg.pinv(Cholesky)
    temp_g = np.matmul(Cholesky_inv, np.matmul(A, Cholesky_inv.T))

    values, vectors = np.linalg.eigh(temp_g)
    H = np.matmul(np.diag(1 / np.sqrt(values)), np.matmul(vectors.T, Cholesky_inv))

    return H.T

def simulate(A, T, P, K, J, D):
    """
    Simulates a model from framework 2.
    Returns the regressor object, F0, G0, Gamma and epsilon
    """
    ## F is autoregressive with matrix A
    e = np.random.randn(K, T)
    for i in range(1, T):
        e[:, i:i+1] += np.matmul(A, e[:, i-1:i])

    F = e.T

    ## X is standard normal 1 covariate only
    X = np.random.randn(P, D)

    ## Phi is generated from three functions
    Phi = np.zeros((P, J))
    Phi[:, 0:1] = X
    Phi[:, 1:2] = X ** 2 - 1
    Phi[:, 2:3] = X ** 3 - 2 * X

    ## Random loadings
    B = np.random.randn(J, K)

    G = np.matmul(Phi, B)

    H = co_diagonalize(np.matmul(F.T, F), np.linalg.pinv(np.matmul(G.T, G)))
    H = np.sqrt(T) * H

    ## Rotate loadings and factors
    F0 = np.matmul(F, H)
    G0 = np.matmul(G, np.linalg.pinv(H.T))

    ## Simulate the noise
    alpha = 7.06
    beta = 536.93
    mu = -0.0019
    sig = 0.1499
    Diag = np.diag(np.random.gamma(alpha, 1 / beta, size=P))
    sigma0 = np.random.normal(mu, sig, size=(P, P))
    for i in range(P):
        sigma0[i, i] = 1.0

    for i in range(P):
        for j in range(i, P):
            sigma0[i, j] = sigma0[j, i]

    sigma0 = sigma0 * (np.abs(sigma0) > 0.03)
    sigma0 = cov_nearest(sigma0, threshold=1e-10)

    covariance = np.matmul(Diag, np.matmul(sigma0, Diag))

    noise = np.random.multivariate_normal(np.zeros(P), covariance, T).T

    ## Finalize the model

    Y = np.matmul(G, F.T) + noise

    reg_PPCA = Regressor()
    # reg_PPCA.add_line([1, 0])
    # reg_PPCA.add_square([1, 0, -1])
    # reg_PPCA.add_cube([1, 0, -2, 0])
    reg_PPCA.add_cubic_splines(X.min(), X.max(), int(3 * (P * min(P, T)) ** (0.25)))

    reg_PPCA.fit_PPCA(Y, X, 3, find_K=False)

    reg_PC = Regressor()
    reg_PC.add_line([1, 0])
    reg_PC.add_square([1, 0, -1])
    reg_PC.add_cube([1, 0, -2, 0])

    reg_PC.fit_PC(Y, 3, find_K=False)

    return reg_PPCA, reg_PC, F0, G0, np.zeros((P, K)), noise

def err(reg, F0, G0):
    """
    Compute the maximal error and the Frobenius error between F0 and F, and G and G

    The model is exactly equivalent if we multiply the same column of F and G by -1,
    and if we apply any same permutation to the columns of F and G.

    Thus, to compute the error, we need to find the permutation and the columns to multiply by -1 
    """
    P, T = reg.Y.shape
    K = F0.shape[1]

    ## Trying every subset of columns to be multiplied by 1, and every permutation of the columns
    good_perm = None
    good_subset = None
    good_error = np.inf
    for perm in permutations(range(K)):
        for k in range(K):
            for subset in combinations(range(K), k+1):
                ones = np.ones_like(reg.F)
                ones[:, subset] *= -1
                F_err_fro = 1 / np.sqrt(T) * np.linalg.norm(F0 - ones * reg.F[:, perm])
                if F_err_fro < good_error:
                    good_perm = perm
                    good_error = F_err_fro
                    good_subset = subset

    ones_F = np.ones_like(reg.F)
    ones_F[:, good_subset] *= -1

    ones_G = np.ones_like(reg.G)
    ones_G[:, good_subset] *= -1
    
    F_err_fro = 1 / np.sqrt(T) * np.linalg.norm(F0 - ones_F * reg.F[:, good_perm])
    F_err_max = np.max(np.abs(F0 - ones_F * reg.F[:, good_perm]))

    G_err_fro = 1 / np.sqrt(P) * np.linalg.norm(G0 - ones_G * reg.G[:, good_perm])
    G_err_max = np.max(np.abs(G0 - ones_G * reg.G[:, good_perm]))

    return F_err_fro, F_err_max, G_err_fro, G_err_max

def generate_curve(P_min, P_max, P_number, T_min, T_max, T_number, A, N=500):

    res_PPCA_F_max = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PC_F_max = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PPCA_F_Fro = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PC_F_Fro = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PPCA_G_max = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PC_G_max = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PPCA_G_Fro = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))
    res_PC_G_Fro = pd.DataFrame(index=np.linspace(P_min, P_max, P_number, dtype=int), columns=np.linspace(T_min, T_max, T_number, dtype=int))

    K = 3
    J = 3
    D = 1

    for P in np.linspace(P_min, P_max, P_number, dtype=int):
        for T in np.linspace(T_min, T_max, T_number, dtype=int):

            li_F_fro = []
            li_F_max = []
            li_G_fro = []
            li_G_max = []
            li_F_fro_PC = []
            li_F_max_PC = []
            li_G_fro_PC = []
            li_G_max_PC = []
            for i in range(N):
                reg_PPCA, reg_PC, F0, G0, Gamma, eps = simulate(A, T, P, K, J, D)
                
                F_err_fro, F_err_max, G_err_fro, G_err_max = err(reg_PPCA, F0, G0)
                li_F_fro.append(F_err_fro)
                li_F_max.append(F_err_max)
                li_G_fro.append(G_err_fro)
                li_G_max.append(G_err_max)
                
                F_err_fro, F_err_max, G_err_fro, G_err_max = err(reg_PC, F0, G0)
                li_F_fro_PC.append(F_err_fro)
                li_F_max_PC.append(F_err_max)
                li_G_fro_PC.append(G_err_fro)
                li_G_max_PC.append(G_err_max)

            li_F_fro = np.array(li_F_fro)
            li_F_max = np.array(li_F_max)
            li_G_fro = np.array(li_G_fro)
            li_G_max = np.array(li_G_max)

            li_F_fro_PC = np.array(li_F_fro_PC)
            li_F_max_PC = np.array(li_F_max_PC)
            li_G_fro_PC = np.array(li_G_fro_PC)
            li_G_max_PC = np.array(li_G_max_PC)

            res_PPCA_F_max.loc[P, T] = li_F_max.mean()
            res_PC_F_max.loc[P, T] = li_F_max_PC.mean()
            res_PPCA_F_Fro.loc[P, T] = li_F_fro.mean()
            res_PC_F_Fro.loc[P, T] = li_F_fro_PC.mean()
            res_PPCA_G_max.loc[P, T] = li_G_max.mean()
            res_PC_G_max.loc[P, T] = li_G_max_PC.mean()
            res_PPCA_G_Fro.loc[P, T] = li_G_fro.mean()
            res_PC_G_Fro.loc[P, T] = li_G_fro_PC.mean()

    return res_PPCA_F_max, res_PC_F_max, res_PPCA_F_Fro, res_PC_F_Fro,\
           res_PPCA_G_max, res_PC_G_max, res_PPCA_G_Fro, res_PC_G_Fro


if __name__ == "__main__":

    ## Design 2, much easier

    A = np.array([[-0.0371, -0.1226, -0.1130],
                   [-0.2339, 0.1060, -0.2793],
                   [0.2803, 0.0755, -0.0529]])

    res_PPCA_F_max, res_PC_F_max, res_PPCA_F_Fro, res_PC_F_Fro, res_PPCA_G_max,\
    res_PC_G_max, res_PPCA_G_Fro, res_PC_G_Fro = generate_curve(20, 20, 1, 10, 50, 10, A, N=500)

    res_PPCA_F_Fro.T.plot()
    res_PC_F_Fro.T.plot()
    plt.show()
    