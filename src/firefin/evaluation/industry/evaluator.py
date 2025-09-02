# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

import typing

import pandas as pd

from ...core.plot import plots
from ...core.eva_utils import ForwardReturns, IC, QuantileReturns, compute_ic, compute_quantile_returns

__all__ = ["Evaluator"]


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Convert the index of a DataFrame to a DatetimeIndex if it is not.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with an index to be checked/converted.

    Returns
    -------
    pd.DataFrame
        A DataFrame with its index converted to DatetimeIndex if needed.
    '''

    out = df.copy(deep = False)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out


class Evaluator:
    def __init__(
            self, 
            factor: pd.DataFrame, 
            forward_returns: ForwardReturns
    ):
        self.factor = factor
        self.forward_returns = forward_returns
        self._to_datetime_index()
        self._reindex_forward_returns()
        self._result = {}

    def _to_datetime_index(self):
        '''
        Convert the index of the factor and forward return DataFrames to DatetimeIndex.
        '''

        self.factor = to_datetime_index(self.factor)
        self.forward_returns = {k: to_datetime_index(v) for k, v in self.forward_returns.items()}

    def _reindex_forward_returns(self):
        '''
        Align the index and columns of all forward return DataFrames with the factor DataFrame, 
        without creating a new object when their structures are already consistent.
        '''

        self.forward_returns = {k: v.reindex_like(self.factor, copy = False) for k, v in self.forward_returns.items()}

    def get_ic(
            self, 
            method: typing.Literal["pearson", "kendall", "spearman"], 
            plot = True
    ) -> IC:
        '''
        Get charts and statistical features of Information Coefficient (IC) between the factor 
        and forward returns.

        This method calculates the IC time series using the specified correlation method for 
        each holding period in `forward_returns`. It also visualizes the IC results using 
        four plots:
            - Line plot of IC over time
            - Cumulative IC curve
            - Histogram of IC distribution
            - QQ plot against the normal distribution

        Additionally, summary statistics will be printed, including:
            - Mean and standard deviation
            - Information Ratio (IR)
            - Proportion of IC > 0 and < 0
            - Proportion of IC > 3%, < 3%
            - Proportion of IC > 5%, < 5%

        Parameters
        ----------
        method : {"pearson", "kendall", "spearman"}
            The correlation method to use for computing IC.
        plot : bool, default=True
            Whether to display plots.

        Returns
        -------
        IC
            A dataframe containing the computed IC time series for each holding period.
        '''

        cache_key = ("ic", (method,))
        if cache_key not in self._result:
            self._result[cache_key] = compute_ic(self.factor, self.forward_returns, method)
        ic = self._result[cache_key]
        if plot:
            plots.plt_ic(ic)
        return ic

    def get_quantile_returns(self, quantiles: int = 5, plot=True) -> QuantileReturns:
        cache_key = ("quantile_returns", (quantiles,))
        if cache_key not in self._result:
            self._result[cache_key] = compute_quantile_returns(self.factor, self.forward_returns, quantiles)
        qt = self._result[cache_key]
        if plot:
            plots.plt_quantile_cumulative_returns(qt)
            plots.plt_quantile_cumulated_end_returns(qt)
        return qt
