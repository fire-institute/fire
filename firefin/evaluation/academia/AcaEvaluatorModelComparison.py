import pandas as pd
from ...core.algorithm.MSR_Test import MSRTest
from ...core.eva_utils import ForwardReturns

class AcaEvaluatorModelComparison:
    def __init__(self, factor1: pd.DataFrame, factor2: pd.DataFrame, forward_returns: ForwardReturns):
        """
        Parameters:
            factor1 & factor2: pd.DataFrame
                Factor exposure data (Time × Stock)
            forward_returns: dict[str, pd.DataFrame]
                A dictionary where each key is a holding period, and the value is a DataFrame of future returns (Time × Stock)
        """

        self.factor1 = factor1
        self.factor2 = factor2
        self.forward_returns = forward_returns

    def run_msr_test(self, regularize=True):
        """
        Compare the Maximum Sharpe Ratios of two factor models using a Z-test.
        Args:
            regularize_covariance (bool): If True, regularize the covariance matrix.
        Returns:
            dict: {
                'msr_a': float,  # MSR of model A
                'msr_b': float,  # MSR of model B
                'test_stat': float,  # Z-statistic
                'p_value': float  # two-sided p-value
            }
        """
        return MSRTest.run_msr_comparison(model_a=self.factor1, model_b=self.factor2, regularize_covariance=True)