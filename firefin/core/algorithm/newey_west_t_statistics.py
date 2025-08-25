import numpy as np
import pandas as pd
import statsmodels.api as sm
from .regression import BatchRegressionResult


def get_sandwich_arrays(residuals, X):
    """
    return
    -------
    xu           : (T, K) or  (T, K*N)
    hessian_inv  : (K, K)  or (K*N, K*N)
    N            : number of equations
    """
    residuals = np.asarray(residuals, dtype=float)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2-d (T, K)")
    T, K = X.shape

    # ---------- single equation ----------
    if residuals.ndim == 1:
        if residuals.shape[0] != T:
            raise ValueError("rows mismatch")
        xu = X * residuals[:, None]              # (T, K)
        hessian_inv = np.linalg.pinv(X.T @ X)    # (K, K)
        N = 1

    # ---------- multi equations ----------
    elif residuals.ndim == 2:
        if residuals.shape[0] != T:
            raise ValueError("rows mismatch")
        N = residuals.shape[1]

        
        xu = np.einsum('ti,tk->tik', residuals, X).reshape(T, K * N)

        # H⁻¹ = I_m ⊗ (X'X)⁻¹
        XtX_inv    = np.linalg.pinv(X.T @ X)
        hessian_inv = np.kron(np.eye(N), XtX_inv)

    else:
        raise ValueError("residuals must be 1-d or 2-d")

    return xu, hessian_inv, N


from statsmodels.stats.sandwich_covariance import S_hac_simple
from statsmodels.stats.sandwich_covariance import _HCCM2

def cov_hac(xu, hessian_inv, N=1,use_correction=True):
    sigma = S_hac_simple(xu)
    cov_hac = _HCCM2(hessian_inv, sigma)
    if use_correction:
        nobs, k_params = xu.shape
        k_each = k_params // N  # =K
        cov_hac *= nobs / float(nobs - k_each)

    return cov_hac
def newey_west_t(residuals, X, beta, use_correction=True):
    """
    residuals : (T,) or (T, N)     ndarray
    X         : (T, K)             ndarray
    beta      : (K,)  or  (K, N)   ndarray

    return:
    t values of each parameter (N*K)
    """
    xu, h_inv, N = get_sandwich_arrays(residuals, X)
    cov = cov_hac(xu, h_inv, N, use_correction=use_correction)
    se = np.sqrt(np.diag(cov))

    beta_vec = np.asarray(beta).reshape(-1, order='F')
    t_values = beta_vec / se
    K = X.shape[1]
    t_values_arr = t_values.reshape(N, K)
    return t_values_arr

class NeweyWestTest:
    @staticmethod
    def newey_west_t_test(result: BatchRegressionResult, X: list[pd.Series]):
        X = [s.fillna(0) for s in X]
        factor_names = ["alpha"] + [s.name for s in X]
        stocks_names = result.alpha.columns
        residuals = result.residuals
        betas = np.array(result.beta).transpose(0,2,1) # K*N*T
        alphas =np.array(result.alpha).T # N*T
        keys = residuals.keys()
        X = np.stack([s.values for s in X],axis=0) # K*T
        t_values = []
        for idx, key in enumerate(keys):
            residual = np.array(residuals[key])
            window = residual.shape[0]
            beta = betas[:,:, window-1+idx]
            alpha = alphas[:, window-1+idx]
            alpha_row = alpha[np.newaxis, :]
            Beta = np.vstack([alpha_row, beta]) # (K+1)*N
            x = sm.add_constant(X[:,idx:idx+window].T) # window*(K+1)
            t_val = newey_west_t(residual, x, Beta)
            t_val_df = pd.DataFrame(data = t_val, index = stocks_names, columns = factor_names )
            t_values.append(t_val_df)
        t_values_dict = dict(zip(keys, t_values))
        return t_values_dict