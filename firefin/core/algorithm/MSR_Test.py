import numpy as np
import pandas as pd
from scipy.stats import norm

class MSRTest:
    """
    A class to compute and statistically compare the Maximum Sharpe Ratios (MSRs)
    between two factor models using the asymptotic test from Barillas & Shanken (2018).
    """

    @staticmethod
    def compute_max_sharpe_ratio(factor_returns: pd.DataFrame, regularize_covariance: bool = False) -> tuple:
        """
        Compute the maximum Sharpe ratio for a factor model.

        Args:
            factor_returns (pd.DataFrame): T × K matrix of factor returns.
            regularize_covariance (bool): If True, regularize the covariance matrix.
        Returns:
            tuple:
                - float: Maximum Sharpe Ratio
                - np.ndarray: Mean return vector (μ)
                - np.ndarray: Covariance matrix (Σ)
        """
        mu = factor_returns.mean().values
        sigma = np.cov(factor_returns.T, ddof=1)
        
        # Regularizing the covariance matrix
        if regularize_covariance:
            epsilon = 1e-6  # small constant for regularization
            sigma += np.eye(sigma.shape[0]) * epsilon

        msr = np.sqrt(mu @ np.linalg.inv(sigma) @ mu)
        return msr, mu, sigma

    @staticmethod
    def asymptotic_variance_msr_squared(mu: np.ndarray, sigma: np.ndarray, T: int) -> float:
        """
        Compute the asymptotic variance of the squared maximum Sharpe ratio.

        Args:
            mu (np.ndarray): Mean return vector.
            sigma (np.ndarray): Covariance matrix.
            T (int): Sample size.

        Returns:
            float: Asymptotic variance of MSR².
        """
        inv_sigma = np.linalg.inv(sigma)
        term = 4 * (mu @ inv_sigma @ sigma @ inv_sigma @ mu)
        return term / T

    @staticmethod
    def run_msr_comparison(model_a: pd.DataFrame, model_b: pd.DataFrame, regularize_covariance: bool = False) -> dict:
        """
        Compare the Maximum Sharpe Ratios of two factor models using a Z-test.

        Args:
            model_a (pd.DataFrame): T × K matrix of factor returns for model A.
            model_b (pd.DataFrame): T × K matrix of factor returns for model B.
            regularize_covariance (bool): If True, regularize the covariance matrix.
        Returns:
            dict: {
                'msr_a': float,  # MSR of model A
                'msr_b': float,  # MSR of model B
                'test_stat': float,  # Z-statistic
                'p_value': float  # two-sided p-value
            }
        """
        T = model_a.shape[0]
        # Compute MSRs and their components
        msr_a, mu_a, sigma_a = MSRTest.compute_max_sharpe_ratio(model_a, regularize_covariance)
        msr_b, mu_b, sigma_b = MSRTest.compute_max_sharpe_ratio(model_b, regularize_covariance)

        # Compute variances of MSR²
        msr2_a = msr_a ** 2
        msr2_b = msr_b ** 2
        var_a = MSRTest.asymptotic_variance_msr_squared(mu_a, sigma_a, T)
        var_b = MSRTest.asymptotic_variance_msr_squared(mu_b, sigma_b, T)

        # Z-test for MSR² difference
        diff = msr2_a - msr2_b
        std_error = np.sqrt(var_a + var_b)
        z_stat = diff / std_error
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

        return {
            "msr_a": msr_a,
            "msr_b": msr_b,
            "test_stat": z_stat,
            "p_value": p_value
        }