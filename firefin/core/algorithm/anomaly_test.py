from __future__ import annotations

import pandas as pd
from typing import List, Optional, Union

from .regression import _regression, RegressionResult
from ..eva_utils import QuantileReturns


class AnomalyTest:
    """
    Perform anomaly tests by regressing portfolio returns on a specified factor model
    and summarizing the resulting parameter estimates and test statistics.

    Attributes
    ----------
    portfolio_returns : pd.DataFrame
        DataFrame of portfolio returns, with each column representing a distinct portfolio.
    factor_model : pd.DataFrame
        DataFrame containing factor return series as independent variables.
    _regression_results : dict[str, RegressionResult]
        Mapping from portfolio name to its fitted RegressionResult.
    """

    def __init__(
        self,
        portfolio_returns: QuantileReturns,
        factor_model: Union[pd.DataFrame, List[pd.Series]],
    ) -> None:
        """
        Initialize the AnomalyTest object.

        Parameters
        ----------
        portfolio_returns : QuantileReturns
            Data structure holding portfolio returns. Must be convertible to a DataFrame
            and have a .columns attribute.
        factor_model : DataFrame or list of Series
            Factor return series used as regressors. Can be a DataFrame or a list of Series.

        Raises
        ------
        TypeError
            If inputs are not of the expected types.
        ValueError
            If factor_model is empty.
        """
        # Convert portfolio_returns to DataFrame if needed
        if not hasattr(portfolio_returns, "columns"):
            raise TypeError("`portfolio_returns` must have a `columns` attribute.")
        self.portfolio_returns = (
            portfolio_returns
            if isinstance(portfolio_returns, pd.DataFrame)
            else pd.DataFrame(portfolio_returns)
        )

        # Build factor DataFrame
        if isinstance(factor_model, list):
            if not factor_model:
                raise ValueError("`factor_model` cannot be an empty list.")
            self.factor_model = pd.concat(factor_model, axis=1)
        elif isinstance(factor_model, pd.DataFrame):
            if factor_model.empty:
                raise ValueError("`factor_model` DataFrame cannot be empty.")
            self.factor_model = factor_model
        else:
            raise TypeError("`factor_model` must be a DataFrame or a list of Series.")
        
        self.factor_model.columns = [f'Factor_{i}' for i in self.factor_model.columns]
        self._regression_results: dict[str, RegressionResult] = {}

    def fit(
        self,
        cov_type: Optional[str] = None,
        cov_kwds: Optional[dict] = None,
    ) -> AnomalyTest:
        """
        Fit time-series regressions of each portfolio return on the factor model.

        Parameters
        ----------
        cov_type : str, optional
            Covariance estimator type (e.g., 'HAC' for Newey–West or 'HC1').
            If None, uses the default homoskedastic standard errors.
        cov_kwds : dict, optional
            Keyword arguments for the covariance estimator (e.g., {'maxlags': L}).

        Returns
        -------
        self : AnomalyTest
            The fitted AnomalyTest instance, with regression results stored internally.
        """
        # Align portfolio and factor data on the same dates
        df_all = pd.concat([self.portfolio_returns, self.factor_model], axis=1).dropna()
        y_df = df_all[self.portfolio_returns.columns]
        x_df = df_all[self.factor_model.columns]

        for port in y_df.columns:
            raw_res = _regression(
                x=x_df,
                y=y_df[port],
                w=None,
                fit_intercept=True,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
            )
            # Determine if regression is univariate
            univariate = (x_df.shape[1] == 1)
            wrapped = RegressionResult(raw_res, fit_intercept=True, univariate=univariate)
            self._regression_results[port] = wrapped
        return self

    def test_statistics(self) -> pd.DataFrame:
        """
        Generate a comprehensive summary table of parameter estimates and test statistics.

        Returns
        -------
        summary : pd.DataFrame
            A MultiIndex DataFrame with rows labeled by (portfolio, parameter) and
            columns ['coef', 'tvalue', 'stderr', 'pvalue'].
        """
        records = []
        for port, res in self._regression_results.items():
            params = res.sm_result.params
            tvals = res.sm_result.tvalues
            stderrs = res.sm_result.bse
            pvals = res.sm_result.pvalues
            records.append({
                'coef': params,
                'tvalue': tvals,
                'stderr': stderrs,
                'pvalue': pvals
            })
        return records

    def alpha(self, portfolio: str) -> float:
        """
        Retrieve the intercept (alpha) from the regression of a specific portfolio.

        Parameters
        ----------
        portfolio : str
            The name of the portfolio column.

        Returns
        -------
        alpha : float
            The estimated intercept term.
        """
        return self._regression_results[portfolio].alpha  # type: ignore

    def betas(self, portfolio: str) -> pd.Series:
        """
        Retrieve the factor loadings (betas) for a specific portfolio.

        Parameters
        ----------
        portfolio : str
            The name of the portfolio column.

        Returns
        -------
        betas : pd.Series
            Series of estimated factor coefficients, indexed by factor name.
        """
        return self._regression_results[portfolio].beta  # type: ignore
