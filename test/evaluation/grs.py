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
    Sigma = np.cov(resid, rowvar=True,ddof=K)#N*N残差协方差矩阵

    # Mean excess returns of the risk factors
    factor_mean = np.mean(factors, axis=1,keepdims=True)#K*1的均值矩阵


    # Covariance matrix of factors
    omega=np.cov(factors,rowvar=True,ddof=-1)
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
