import pandas as pd
import numpy as np
from scipy.stats import f

def grs_test(resid: pd.DataFrame, alpha: pd.DataFrame, date , window_size, factors:pd.DataFrame ,label: str = "tab:grs",caption: str = "GRS 检验结果") -> None:
    """ Perform the Gibbons, Ross and Shanken (1989) test.
        :param resid: Matrix of residuals from the OLS of size TxN.
        :param alpha: Vector of alphas from the OLS of size Nx1.
        :param factors: Matrix of factor returns size KxT.
        :return: Test statistic and p-value of the test statistic.
    """
    #数据处理
    alpha = alpha.loc[date].to_numpy()
    if alpha.ndim == 1:
        alpha = alpha.reshape(1, -1).T #N*1
    rows = []
    for i in range(window_size):
        resid_i = resid[i].loc[date].to_numpy()
        rows.append(resid_i)
    resid = np.stack(rows, axis=0).T
    factors = factors.loc[:date].tail(window_size).to_numpy()
    if factors.ndim == 1:
        factors = factors.reshape(1, -1)


    # Determine the time series and assets
    N, T = resid.shape
    K= factors.shape[0]  # factors是K*T矩阵
    try:
        T-N-K >0
    except ValueError as e:
        print(f"time period should be greater than number of assets{e}")

    # Covariance of the residuals
    Sigma = np.cov(resid, rowvar=True,ddof=K+1)#N*N残差协方差矩阵

    # Mean excess returns of the risk factors
    factor_mean = np.mean(factors, axis=1,keepdims=True)#K*1的均值矩阵


    # Covariance matrix of factors
    omega=np.cov(factors,rowvar=True,ddof=0)
    omega = np.atleast_2d(omega)
    inv_omega = np.linalg.pinv(omega)
    inv_Sigma= np.linalg.pinv(Sigma)
    mult_=(factor_mean.T @ inv_omega @ factor_mean).item()
    mult=1/(1+mult_)
    inter=(alpha.T @ inv_Sigma @ alpha).item()
    # GRS statistic
    dTestStat = (T / N) * ((T - N - K) / (T - K - 1)) * inter * mult
    # p-value of the F-test
    df1=N
    df2=T-N-K
    pVal = 1 - f.cdf(dTestStat, df1, df2)
    df = pd.DataFrame(
        {"Value": [dTestStat, pVal]},
        index=["GRS 统计量", "p‑value"],
    )

    # 打印 LaTeX 代码
    print(df.to_latex(
        float_format="%.4f",
        caption=caption,
        label=label,
        header=False
    ))


def _grs(resid,
         alpha: pd.DataFrame | pd.Series | np.ndarray,
         factors):
    """
    Gibbons, Ross & Shanken (1989) GRS 检验（无截面/滚动窗口版）。
    resid   : T×N 的 DataFrame 或 ndarray  — OLS 残差
    alpha   : 长度 N 的 Series、1D ndarray 或 N×1 ndarray
    factors : T×K 的 DataFrame 或 ndarray  — 因子收益
    Returns: pvalue
    """

    resid_arr = np.asarray(resid)
    if resid_arr.ndim != 2:
        raise ValueError(f"resid must be 2d (T, N)，receive ndim={resid_arr.ndim}")
    resid_np = resid_arr.T                     # N×T
    N, T = resid_np.shape

    alpha_arr = np.asarray(alpha).reshape(-1, 1)  # N×1
    if alpha_arr.shape[0] != N:
        raise ValueError("alpha must has the same length N as resid")

    factors_arr = np.asarray(factors)
    if factors_arr.ndim != 2:
        raise ValueError(f"factors must be 2d (T, K)，receive ndim={factors_arr.ndim}")
    if factors_arr.shape[0] != T:
        raise ValueError("row length of factors must be equal to T")
    factors_np = factors_arr.T                 # K×T
    K = factors_np.shape[0]

    if T <= N + K:
        raise ValueError(f"T <= N+K, this is not allowed")

    Sigma = np.cov(resid_np,    rowvar=True, ddof=K+1)
    Omega = np.cov(factors_np,  rowvar=True, ddof=0)
    Sigma = np.atleast_2d(Sigma)
    Omega = np.atleast_2d(Omega)


    eps = 1e-12
    if Sigma.shape[0] == Sigma.shape[1]:
        Sigma = Sigma + eps * np.eye(Sigma.shape[0])
    if Omega.shape[0] == Omega.shape[1]:
        Omega = Omega + eps * np.eye(Omega.shape[0])

    inv_Omega = np.linalg.pinv(Omega)
    inv_Sigma = np.linalg.pinv(Sigma)

    mu_f  = factors_np.mean(axis=1, keepdims=True)        # K×1
    mult  = 1.0 / (1.0 + (mu_f.T @ inv_Omega @ mu_f).item())
    inter = (alpha_arr.T @ inv_Sigma @ alpha_arr).item()
    grs_stat = (T / N) * ((T - N - K) / (T - K - 1)) * inter * mult

    dof1, dof2 = N, T - N - K
    p_val = 1 - f.cdf(grs_stat, dof1, dof2)
    return p_val