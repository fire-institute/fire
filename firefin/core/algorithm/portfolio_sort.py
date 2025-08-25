"""
Portfolio Sort Implementation for Academic Research
---------------------------------------------------
This module provides a class for performing single and double portfolio sorts
based on characteristics, market capitalization, and returns. The implementation
focuses on clarity, documentation, and best practices for financial research.
"""

import typing
import numpy as np
import pandas as pd
from .newey_west_ttest_1samp import NeweyWestTTest
from ..eva_utils import factor_to_quantile, factor_to_quantile_dependent_double_sort,double_sort_returns_independent
from ..eva_utils import _compute_quantile_df, _compute_weighted_quantile_df
from ..eva_utils import ForwardReturns, QuantileReturns

StatisticResults = typing.NewType("StatisticResults", dict[str, pd.DataFrame])

class PortfolioSort:
    """
    Class to perform single and double portfolio sorts based on characteristics.
    """

    @staticmethod
    def single_sort(
        factor: pd.DataFrame,
        forward_returns: ForwardReturns,
        quantiles: int,
        value_weighted: bool = True,
        get_quantile_sorts: bool = False,
        market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Perform single portfolio sort based on characteristic and create long-short portfolio.
        
        Args:
            factor: (Time x Stock) DataFrame of characteristic exposures
            forward_returns: period : (Time x Stock) DataFrame of returns
            market_cap: (Time x Stock) DataFrame of market capitalizations
            quantiles: number of quantiles
            value_weighted: Use market cap weighting (True) or equal weighting (False)
            get_quantile_sorts: Return portfolio assignments
        Returns:
            Portfolio returns and statistical results
        """
        # 1. DATA PREPARATION
        # assume factor, forward_return, market_cap are aligned DataFrames in our case
        # 2. QUANTILE CALCULATIONS
        quantile_sorts = factor_to_quantile(factor, quantiles)

        # Early exit if quantile assignments requested
        if get_quantile_sorts:
            return quantile_sorts
        
        # 3. RETURN CALCULATIONS
        # TODO: Add support for other weighting schemes
        # TODO: Add transaction costs
        if value_weighted:
            portfolio_returns = QuantileReturns ({
                period: _compute_weighted_quantile_df(quantile_sorts, period_returns, market_cap,quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })
        else:
            # equal weighted
            portfolio_returns = QuantileReturns ({
                period: _compute_quantile_df(quantile_sorts, period_returns, quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })

        # 4. HEDGE PORTFOLIO (High-Low)
        for period, _ in forward_returns.items():
            portfolio_returns[period]["H-L"] = (
                portfolio_returns[period][quantiles] - portfolio_returns[period][1]
            )
    
        return portfolio_returns

    @staticmethod
    def double_sort(
        factor1: pd.DataFrame,
        factor2: pd.DataFrame,
        forward_returns: ForwardReturns,
        quantiles: typing.Tuple[int, int] = (5, 5),
        dependent: bool = False,
        value_weighted: bool = True,
        get_quantile_sorts: bool = False,
        market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Sort securities based on two characteristics.

        Args:
            factor1: (Time x Stock) DataFrame of characteristic exposures
            factor2: (Time x Stock) DataFrame of characteristic exposures
            forward_returns: period : (Time x Stock) DataFrame of returns
            market_cap: (Time x Stock) DataFrame of market capitalizations
            dependent: Whether to use dependent sorting (True) or independent sorting (False)
            quantiles: number of quantiles
            value_weighted: Use market cap weighting (True) or equal weighting (False)
            get_quantile_sorts: Return portfolio assignments
        Returns:
            Portfolio returns and statistical results
        """
        # Ensure that factor1 and factor2 have the same index and columns
        assert factor1.index.equals(factor2.index) and factor1.columns.equals(factor2.columns), \
            "factor1 and factor2 must have the same index and columns"

        # 1. DATA PREPARATION
        # assume factor1, factor2, forward_return, market_cap are aligned DataFrames in our case
        # 2. QUANTILE CALCULATIONS
        if dependent:
            # Dependent sorting (conditional double sorting)
            """
            Note from Professor SHI:

            Suppose we first sort the stocks based on X1, dividing all stocks into L1 groups. Then, within each of
            these L1 groups, we further sort the stocks based on X2, dividing the stocks into L2 groups. Again, a
            total of L1 × L2 groups

            The two sorting variables are NOT treated equally: the first sorting variable acts solely as a control
            variable, and the main interest is the relationship between the second sorting variable and asset
            returns. A factor should only be constructed based on the second sorting variable

            Lets assume factor1 is the control variable and factor2 is the main variable of interest.
            We will first sort the stocks based on factor1, then within each group, we will sort the stocks based on factor2.
            """

            combined_sorts =  factor_to_quantile_dependent_double_sort(factor1, factor2, quantiles)
        else:
            # independent sorting (unconditional double sorting)
            # Independent sorting will result some NONE quantile

            quantile_sorts_factor1 = factor_to_quantile(factor1, quantiles[0]).astype(int)
            quantile_sorts_factor2 = factor_to_quantile(factor2, quantiles[1]).astype(int)
            # quantile_sorts to string and add them to q1_q2 format
            combined_sorts = quantile_sorts_factor1.astype(str) + "_" + quantile_sorts_factor2.astype(str)

        # Initialize a dictionary to store the portfolio returns
        portfolio_returns = {}

        # 3. RETURN CALCULATIONS
        for period, period_returns in forward_returns.items():

            # Calculate returns for each combined quantile
            if value_weighted:
                period_portfolio_returns = _compute_weighted_quantile_df(
                    combined_sorts, period_returns, market_cap, reindex=False, quantiles= quantiles[0] * quantiles[1]
                )
            else:
                # equal weighted
                period_portfolio_returns = _compute_quantile_df(
                    combined_sorts, period_returns, reindex=False, quantiles= quantiles[0] * quantiles[1]
                )
            # Store the results
            portfolio_returns[period] = period_portfolio_returns

        # 4. HEDGE PORTFOLIO (High-High vs Low-Low)
        for period, _ in forward_returns.items():
            high_high = portfolio_returns[period].xs(f"{quantiles[0]}_{quantiles[1]}", axis=1)
            low_low = portfolio_returns[period].xs("1_1", axis=1)
            portfolio_returns[period]['HH-LL'] = high_high - low_low

        # Early exit if quantile assignments requested
        if get_quantile_sorts:
            return combined_sorts

        return QuantileReturns(portfolio_returns)
    
    @staticmethod
    def double_sort_return_adj_academic(
            size: pd.DataFrame,
            x: pd.DataFrame,
            return_adj: pd.DataFrame,
            value_weighted: bool = True,
            market_cap: pd.DataFrame | None = None,
            size_break: float = 0.5,
            x_breaks: typing.Tuple[float, float] = (0.3, 0.7),
            lag_weights: bool = True,
    ):
        """
        Academic independent 2×3 (Size=50%, X=30/70) with DAILY breakpoints.
        return QuantileReturns {0: DataFrame}；列为 '1_1'..'2_3'。
        """
        portfolio_returns = double_sort_returns_independent(
            size=size,
            x=x,
            return_adj=return_adj,
            value_weighted=value_weighted,
            market_cap=market_cap,
            size_break=size_break,
            x_breaks=x_breaks,
            lag_weights=lag_weights,
        )
        return QuantileReturns({0: portfolio_returns})


    @staticmethod
    def single_sort_return_adj(
            factor: pd.DataFrame,
            return_adj: pd.DataFrame,
            quantiles: int = 5,
            value_weighted: bool = True,
            get_quantile_sorts: bool = False,
            market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Perform a single portfolio sort and compute average returns for each
        quantile using an already‑aligned `return_adj` matrix (Time × Stock).

        Parameters
        ----------
        factor : DataFrame
            Characteristic exposures, shape (Time × Stock).
        return_adj : DataFrame
            Adjusted returns, shape (Time × Stock).
        quantiles : int
            Number of quantile groups.
        value_weighted : bool, default True
            Use value‑weighted (market‑cap) returns if True; equal‑weighted otherwise.
        get_quantile_sorts : bool, default False
            If True, return only the quantile assignment DataFrame.
        market_cap : DataFrame | None
            Market capitalisations, required when `value_weighted=True`.

        Returns
        -------
        QuantileReturns
            key=0
            Columns are 1‥Q quantile returns plus “H‑L”; rows are dates.
        """
        # 1. Assign stocks to quantiles
        quantile_sorts = factor_to_quantile(factor, quantiles)

        if get_quantile_sorts:
            return quantile_sorts

        # 2. Compute portfolio returns
        if value_weighted:
            portfolio_returns = _compute_weighted_quantile_df(
                quantile_sorts,
                return_adj,
                market_cap,
                quantiles=quantiles,
            )
        else:
            portfolio_returns = _compute_quantile_df(
                quantile_sorts,
                return_adj,
                quantiles=quantiles,
            )

        # 3. High–Low hedge portfolio
        portfolio_returns["H-L"] = (
                portfolio_returns[quantiles] - portfolio_returns[1]
        )

        return QuantileReturns({0: portfolio_returns})
    

#2. Double sort based on return_adj
  
    @staticmethod
    def double_sort_return_adj(
            factor1: pd.DataFrame,
            factor2: pd.DataFrame,
            return_adj: pd.DataFrame,
            quantiles: tuple[int, int] = (5, 5),
            dependent: bool = False,
            value_weighted: bool = True,
            get_quantile_sorts: bool = False,
            market_cap: pd.DataFrame | None = None,
    ) -> typing.Union[QuantileReturns, pd.DataFrame]:
        """
        Perform a double portfolio sort on two characteristics and compute
        average returns for each (q1, q2) group using `return_adj`.

        Parameters
        ----------
        factor1, factor2 : DataFrame
            Characteristic exposures, shape (Time × Stock). Must share identical
            index and columns.
        return_adj : DataFrame
            Adjusted returns, shape (Time × Stock).
        quantiles : tuple[int, int], default (5, 5)
            Number of groups for (factor1, factor2).
        dependent : bool, default False
            If True, use dependent (conditional) sorting; otherwise independent.
        value_weighted : bool, default True
            Use value‑weighted (market‑cap) returns if True; equal‑weighted otherwise.
        get_quantile_sorts : bool, default False
            If True, return only the combined quantile assignment DataFrame.
        market_cap : DataFrame | None
            Market capitalisations, required when `value_weighted=True`.

        Returns
        -------
        QuantileReturns
            key=0
            Columns are “q1_q2” groups plus “HH‑LL”; rows are dates.
        """
        assert factor1.index.equals(factor2.index) and factor1.columns.equals(
            factor2.columns
        ), "factor1 and factor2 must have the same index and columns"

        # 1. Assign stocks to combined quantiles
        if dependent:
            combined_sorts = factor_to_quantile_dependent_double_sort(
                factor1, factor2, quantiles
            )
        else:
            q1 = factor_to_quantile(factor1, quantiles[0]).astype(int)
            q2 = factor_to_quantile(factor2, quantiles[1]).astype(int)
            combined_sorts = q1.astype(str) + "_" + q2.astype(str)

        if get_quantile_sorts:
            return combined_sorts

        # 2. Compute portfolio returns
        if value_weighted:
            portfolio_returns = _compute_weighted_quantile_df(
                combined_sorts,
                return_adj,
                market_cap,
                reindex=False,
                quantiles=quantiles[0] * quantiles[1],
            )
        else:
            portfolio_returns = _compute_quantile_df(
                combined_sorts,
                return_adj,
                reindex=False,
                quantiles=quantiles[0] * quantiles[1],
            )

        # 3. High‑High minus Low‑Low hedge
        hh_label = f"{quantiles[0]}_{quantiles[1]}"
        portfolio_returns["HH-LL"] = portfolio_returns[hh_label] - portfolio_returns[
            "1_1"
        ]

        return QuantileReturns({0: portfolio_returns})
    
    @staticmethod
    def get_statistics(result: QuantileReturns, quantiles: int) -> StatisticResults:
        """
        Compute statistical results for single portfolio sort.

        TODO: 
            1. Add more statistics
            2. plot the results
        """        
        # T-Test for all periods
        # periods * (quantiles + H-L)
        t_stats = np.empty((len(result), quantiles + 1), dtype=float)
        p_values = np.empty((len(result), quantiles + 1), dtype=float)
        se_values = np.empty((len(result), quantiles + 1), dtype=float)
        mean_returns = np.empty((len(result), quantiles + 1), dtype=float)
                                
        for n, (_, period_returns) in enumerate(result.items()):
            # T-Test for all periods
            t_stats[n], p_values[n], se_values[n] = np.apply_along_axis(
                NeweyWestTTest.newey_west_ttest_1samp,
                axis=0,
                arr=period_returns,
                popmean=0,
                lags=6,
                nan_policy='omit'
            )
            # other statistics can be added here
            mean_returns[n] = np.nanmean(period_returns, axis=0)

        return StatisticResults({'t_stats': pd.DataFrame(t_stats, index=result.keys(), columns=period_returns.columns),
                'p_values': pd.DataFrame(p_values, index=result.keys(), columns=period_returns.columns),
                'se_values': pd.DataFrame(se_values, index=result.keys(), columns=period_returns.columns),
                'mean_returns': pd.DataFrame(mean_returns, index=result.keys(), columns=period_returns.columns)})