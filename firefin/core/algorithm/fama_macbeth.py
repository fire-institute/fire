import numpy as np
import pandas as pd
from ...core.algorithm.newey_west_ttest_1samp import NeweyWestTTest
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult

## 已改好 增加返回tvalue (原始论文做法）
## 已改好 factor改成list of DataFrame 中间转换成ndarray
class FamaMacBeth:

    @staticmethod
    def run_regression(
        factor: list[pd.Series], return_adj: pd.DataFrame, window: int = 252, n_jobs=4, verbose: int = 0
    ) :
        """
        Run Fama-MacBeth regression.
        """
        if not isinstance(return_adj, pd.DataFrame):
            raise ValueError("return_adj must be a pandas DataFrame.")

        factor_arr = np.stack([s.values for s in factor], axis=0)
        N = return_adj.shape[1]
        factor_3d = np.repeat(factor_arr[:, :, None], N, axis=2)
        # Note: Calculate excess returns if necessary
        # return_adj = return_adj - risk_free_rate
        # excess return is different in many cases, we leave it to the user to handle this.

        # First step: Time-series regressions
        r = RollingRegressor(factor_3d, return_adj, None, fit_intercept=True).fit(window, n_jobs=n_jobs, verbose=verbose)
        # Second step: Cross-sectional regressions
        # This step involves regressing the time-series regression coefficients on the factors
        betas = []
        alphas = []
        for tau in range(window):
            ret_prime = return_adj.shift(tau)
            r_tau = RollingRegressor(r.beta, ret_prime, None, fit_intercept=True).fit(window=None, axis=1, n_jobs=n_jobs, verbose=verbose)
            betas.append(r_tau.beta)
            alphas.append(r_tau.alpha)

        lambda_sum_df = sum(betas)
        alpha_sum_df = sum(alphas)

        arr_betas = np.stack([df.values for df in betas], axis=0)  # 形状: (window, n_rows, n_cols)
        std_values = arr_betas.std(axis=0, ddof=1)
        arr_alphas = np.stack([df.values for df in alphas], axis=0)
        std_values_alphas = arr_alphas.std(axis=0, ddof=1)

        lambda_std_df = pd.DataFrame(std_values, index=betas[0].index, columns=betas[0].columns)
        alphas_std_df = pd.Series(std_values_alphas, index=alphas[0].index)

        lambda_mean_df = lambda_sum_df/window
        alphas_mean_df = alpha_sum_df/window

        t_values = lambda_mean_df / (lambda_std_df/np.sqrt(window))
        t_values_alphas = alphas_mean_df / (alphas_std_df/np.sqrt(window))

        return BatchRegressionResult(beta=lambda_mean_df, alpha=alphas_mean_df, tvalue=t_values, alpha_t=t_values_alphas)

