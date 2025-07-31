import pandas as pd
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult
import math
import numpy as np
#已完成 factor这里输入变成标准的list of series 不要dataframe了
#已完成 增加返回residuals, r2, r2_adj
def cross_sectional_regression(
        factor: list[pd.Series], return_adj: pd.DataFrame, window: int = 252, cov_type : str | None =None, n_jobs=4, verbose: int = 0
) -> BatchRegressionResult:
    """
    Run cross sectional  regression.
    input:
    factor : pd.DataFrame | pd.Series
    return_adj : pd.DataFrame
    window : int
    cov_type : "HAC" for newey west estimation or None for OLS

    return:
    risk premium : result.beta
    anomaly : result.alpha
    anomaly t statistics : result.alpha_t
    """

    if not isinstance(return_adj, pd.DataFrame) and not isinstance(return_adj, pd.Series):
        raise ValueError("return_adj must be a pandas DataFrame.")

    factor_arr = np.stack([s.values for s in factor], axis=0)
    N = return_adj.shape[1]
    factor_3d = np.repeat(factor_arr[:, :, None], N, axis=2)
    # Note: Calculate excess returns if necessary
    # return_adj = return_adj - risk_free_rate
    # excess return is different in many cases, we leave it to the user to handle this.

    # First step: Time-series regressions
    r = RollingRegressor(factor_3d, return_adj, None, fit_intercept=True).fit(window, n_jobs=n_jobs, verbose=verbose)
    T= return_adj.shape[0]
    N= return_adj.shape[1]
    mean_return_df = return_adj.rolling(window).mean()
    if cov_type is None:
        m=RollingRegressor(x=r.beta, y=mean_return_df, w=None, fit_intercept=True).fit(window=None, axis=1,n_jobs=n_jobs, verbose=verbose)
    else:
        optimal_lags = math.floor(4 * (N / 100) ** (2 / 9)) #maxlags refers to Newey, West(1994)
        m=RollingRegressor(x=r.beta, y=mean_return_df, w=None, fit_intercept=True).fit(window=None, axis=1,cov_type= cov_type, cov_kwds={'maxlags': optimal_lags},n_jobs=n_jobs, verbose=verbose)

    return BatchRegressionResult(beta=m.beta, tvalue=m.tvalue,alpha=m.alpha, alpha_t=m.alpha_t, residuals=m.residuals, r2=m.r2, r2_adj=m.r2_adj)
