from .AcaEvaluatorModel import AcaEvaluatorModel
from typing import Literal
from ...core.algorithm.regression import RollingRegressor
import pandas as pd
from ...core.algorithm.grs import _grs
import numpy as np
from .AcademicFactors import bundle_ff3, bundle_ff5, bundle_capm, bundle_ff3_mom
from ...core.plot.table_and_latex_code import latex_table
from ...core.plot.plots import plot_grs_pval, plot_cumulative_alpha


class AcaIndirectEvaluator():

    def __init__(self,
                 factor_portfolio: list[pd.Series],
                 return_adj: pd.DataFrame| None = None,
                 risk_free_rate: pd.Series| None = None,
                 stock_size:pd.DataFrame| None = None, #market cap
                 stock_value:pd.DataFrame| None = None, #bm value
                 op:pd.DataFrame| None = None, #operating profitability
                 ag:pd.DataFrame| None = None, #asset growth
                 mom_signal:pd.DataFrame| None = None,
                 n_jobs: int = 10,
                 verbose: int = 0):

        self.factor_portfolio = factor_portfolio
        self.return_adj = return_adj
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.risk_free_rate = risk_free_rate
        self.stock_size = stock_size
        self.stock_value = stock_value
        self.op = op
        self.ag = ag
        self.mom_signal = mom_signal


    def evaluate_by_other_factors(
            self,
            mode: Literal["capm", "ff3", "ff3_mom", "ff5", "customize"] = "capm",
            factor2: list[pd.Series] | None = None,
            cov_type: str | None = None,
            cov_kwds: dict | None = None,
    ):

        """
            Evaluate the relationship between the portfolio factor returns and a set of
            predefined or custom risk factors using rolling regressions.

            This method applies different factor models (CAPM, Fama–French 3-factor,
            Fama–French 3-factor with momentum, Fama–French 5-factor, or a custom set
            of user-supplied factors) to estimate factor loadings, t-statistics, and
            adjusted R² for each factor portfolio stored in `self.factor_portfolio`.

            Parameters
            ----------
            mode : {"capm", "ff3", "ff3_mom", "ff5", "customize"}, default "capm"
                The factor model to use:
                - "capm"       : Capital Asset Pricing Model (Market factor only)
                - "ff3"        : Fama–French 3-factor model
                - "ff3_mom"    : Fama–French 3-factor model with momentum factor
                - "ff5"        : Fama–French 5-factor model
                - "customize"  : Use user-specified factors from `factor2`
            factor2 : list of pd.Series, optional
                A list of factor return series to be used only when `mode="customize"`.
                Each series should have a datetime index aligned with portfolio returns.
            cov_type: str | None, optional
            The covariance estimator, default is None.
            - If None: use the default homoskedastic standard errors.
            - If "HAC": Newey–West heteroskedasticity-and-autocorrelation robust SE.
            - Other options supported by statsmodels (e.g. "HC0", "HC1", …).
            cov_kwds: dict | None, optional
                The keyword arguments for the covariance estimator, default is None.
                For Newey–West, you’d typically pass `{"maxlags": L}` to control lag length.

            Returns
            -------
            coefficient_df : pd.DataFrame
                Final regression coefficients for each portfolio (last observation),
                including alpha and factor betas.
                Index: portfolio names, Columns: ["alpha", factor names]
            statistics_df : pd.DataFrame
                Final t-statistics for each coefficient (last observation).
                Index: portfolio names, Columns: ["alpha", factor names]
            r2_adj_series : pd.Series
                Adjusted R² values for each portfolio regression.
                Index: portfolio names, Name: "r2_adj"

            """

        factor_names = [s.name for s in self.factor_portfolio ]

        if self.stock_value is None and mode == "ff3":
            raise ValueError("You must provide stock_value when switch mode ff3.")
        elif (self.mom_signal is None) and mode == "ff3_mom":
            raise ValueError("You must provide mom_signal when switch mode ff3_mom.")
        elif (self.ag is None or self.op is None) and mode == "ff5":
            raise ValueError("You must provide ag and op when switch mode ff5.")

        if mode == "capm":
            customized_factor_list = [s.fillna(0) for s in bundle_capm(stock_return=self.return_adj,
                                                                       market_cap=self.stock_size,
                                                                       risk_free_rate=self.risk_free_rate)]
            k = 1

        elif mode == "ff3":
            customized_factor_list = [s.fillna(0) for s in bundle_ff3(stock_return=self.return_adj,
                                                                      size=self.stock_size,
                                                                      book_to_market= self.stock_value,
                                                                      market_cap=self.stock_size,
                                                                      risk_free_rate=self.risk_free_rate)]
            k = 3

        elif mode == "ff3_mom":
            customized_factor_list = [s.fillna(0) for s in bundle_ff3_mom(stock_return=self.return_adj,
                                                                      size=self.stock_size,
                                                                      book_to_market=self.stock_value,
                                                                      momentum_signal=self.mom_signal,
                                                                      market_cap=self.stock_size,
                                                                      risk_free_rate=self.risk_free_rate)]
            k = 4

        elif mode == "ff5":
            customized_factor_list = [s.fillna(0) for s in bundle_ff5(stock_return=self.return_adj,
                                                                      size=self.stock_size,
                                                                      book_to_market=self.stock_value,
                                                                      profitability=self.op,
                                                                      investment=self.ag,
                                                                      market_cap=self.stock_size,
                                                                      risk_free_rate=self.risk_free_rate)]
            k = 5

        elif mode == "customize":
            customized_factor_list = [s.fillna(0) for s in factor2]
            k = len(factor2)

        else:
            raise ValueError(f"unknown mode '{mode}'")

        coefficient = []
        statistics = []
        r2_adj = []

        for factor_portfolio in self.factor_portfolio:
            factor_excess_ret = factor_portfolio.fillna(0)

            customized_factor_arr = np.stack(
                [f.values.reshape(-1, 1) for f in customized_factor_list], axis=0
            )
            res = RollingRegressor(x=customized_factor_arr, y=factor_excess_ret).fit(window=None,cov_type=cov_type,cov_kwds=cov_kwds)


            col_names = ["alpha"] + [s.name for s in customized_factor_list]

            alpha_last = float(res.alpha.iloc[-1])
            beta_last = res.beta.iloc[:, 0].values

            coefficient_df = pd.DataFrame([[alpha_last, *beta_last]], columns=col_names)

            alpha_t_last = float(res.alpha_t.iloc[-1])
            t_last = res.tvalue.iloc[:, 0].values

            statistics_df = pd.DataFrame([[alpha_t_last, *t_last]], columns=col_names)

            n = len(factor_excess_ret)
            X = np.column_stack([f.values for f in customized_factor_list])
            beta_vec = beta_last.reshape(-1, 1)
            alpha_val = alpha_last
            y = factor_excess_ret.values.reshape(-1, 1)

            resid = y - X @ beta_vec - alpha_val
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            R_sqr = 1 - ss_res / ss_tot
            R_sqr_adj = 1 - (1 - R_sqr) * (n - 1) / (n - k - 1)

            coefficient.append(coefficient_df)
            statistics.append(statistics_df)
            r2_adj.append(R_sqr_adj)

        coefficient_df = pd.concat(coefficient)
        statistics_df = pd.concat(statistics)
        coefficient_df.index = factor_names
        statistics_df.index = factor_names
        r2_adj_series = pd.Series(r2_adj, index=factor_names, name="r2_adj")

        return coefficient_df, statistics_df, r2_adj_series


    def cumulated_alpha(
            self,
            mode: Literal["capm", "ff3", "ff3_mom","ff5", "customize"] = "capm",
            factor2: list[pd.Series] | None = None,
            starting_point: int = 20,
            plt : bool = True,
    ):

        """
            Calculate and optionally plot the cumulative alpha of each factor portfolio
            over time based on a specified asset pricing model.

            This method sequentially estimates alpha values from time 0 up to each point
            in time (starting from `starting_point` observations) using full-sample regressions
            of factor portfolio returns on chosen risk factors. The cumulative alpha series
            shows how alpha evolves as more data is included in the regression.

            Parameters
            ----------
            mode : {"capm", "ff3", "ff3_mom", "ff5", "customize"}, default "capm"
                The factor model to use:
                - "capm"       : Capital Asset Pricing Model
                - "ff3"        : Fama–French 3-factor model
                - "ff3_mom"    : Fama–French 3-factor model with momentum factor
                - "ff5"        : Fama–French 5-factor model
                - "customize"  : Use user-specified factors from `factor2`
            factor2 : list of pd.Series, optional
                Required only when `mode="customize"`. A list of factor return series
                with datetime index aligned to portfolio returns.
            starting_point : int, default 20
                Minimum number of observations before starting alpha calculation.
                Cumulative alpha is computed from this point onwards.
            plt : bool, default True
                If True, plot the cumulative alpha series for each portfolio using
                `plot_cumulative_alpha`. If False, return the series instead.

            Returns
            -------
            list of pd.Series or None
                - If `plt=False`: a list of cumulative alpha series, each named after
                  its corresponding factor portfolio, indexed by date.
                - If `plt=True`: no return value; plots are displayed instead.

            """

        if self.stock_value is None and mode == "ff3":
            raise ValueError("You must provide stock_value when switch mode ff3.")
        elif (self.mom_signal is None) and mode == "ff3_mom":
            raise ValueError("You must provide mom_signal when switch mode ff3_mom.")
        elif (self.ag is None or self.op is None) and mode == "ff5":
            raise ValueError("You must provide ag and op when switch mode ff5.")


        if mode == "capm":

            factor_list = [s.fillna(0) for s in bundle_capm(stock_return=self.return_adj, market_cap=self.stock_size, risk_free_rate=self.risk_free_rate)]


        elif mode == "ff3":
            factor_list = [s.fillna(0) for s in bundle_ff3(stock_return=self.return_adj,
                                                           size=self.stock_size,
                                                           book_to_market= self.stock_value,
                                                           market_cap=self.stock_size,
                                                           risk_free_rate=self.risk_free_rate)]

        elif mode == "ff3_mom":
            factor_list = [s.fillna(0) for s in bundle_ff3_mom(stock_return=self.return_adj,
                                                                      size=self.stock_size,
                                                                      book_to_market=self.stock_value,
                                                                      momentum_signal=self.mom_signal,
                                                                      market_cap=self.stock_size,
                                                                      risk_free_rate=self.risk_free_rate)]

        elif mode == "ff5":
            factor_list = [s.fillna(0) for s in bundle_ff5(stock_return=self.return_adj,
                                                           size=self.stock_size,
                                                           book_to_market=self.stock_value,
                                                           profitability=self.op,
                                                           investment=self.ag,
                                                           market_cap=self.stock_size,
                                                           risk_free_rate=self.risk_free_rate)]


        elif mode == "customize" and factor2 is not None:
            factor_list = [s.fillna(0) for s in factor2]

        elif mode == "customize" and factor2 is None:
            raise ValueError("'factor2' is required for 'customize'")

        else:
            raise ValueError(f"unknown mode '{mode}'")

        cumulative_alpha =[]
        for factor in self.factor_portfolio:
            name = factor.name
            factor_portfolio = factor.fillna(0)

            T = len(factor_portfolio)

            cumulated_alpha = []
            for i in range(starting_point, T + 1):

                factor_list_i = [s.iloc[:i] for s in factor_list]

                factor_arr_i = np.stack(
                    [f.values.reshape(-1, 1) for f in factor_list_i], axis=0
                )

                y_i = factor_portfolio.iloc[:i]
                res = RollingRegressor(x=factor_arr_i, y=y_i).fit(window=None)

                alpha_i = float(res.alpha.iloc[-1])
                cumulated_alpha.append(alpha_i)

            alpha_index = factor_portfolio.index[(starting_point-1) : ]
            cumulated_alpha_series = pd.Series(cumulated_alpha, index=alpha_index,name = name)

            cumulated_alpha_series = cumulated_alpha_series.reindex(factor_portfolio.index)
            cumulative_alpha.append(cumulated_alpha_series)
        if plt:
            for s in cumulative_alpha:
                plot_cumulative_alpha(s,title=s.name)
        else:
            return cumulative_alpha

    def evaluate_stability(self,
                           value_weighted: bool = True,
                           mode: Literal["single", "capm", "ff3", "ff3_mom", "ff5"] = "single",
                           window=30,plt:bool = True
                           ):
        """
        this function checks the time series robustness of a factor, using rolling grs test.
        4 modes you can choose to evaluate the stability of this factor
        "single" : regress given return(forward_returns) using this factor, then return the time series grs pvalue
        "capm" : regress given return(forward_returns) using market factor and this factor, then return the time series grs pvalue
        "ff3" :  regress given return(forward_returns) using ff3 factors and this factor, then return the time series grs pvalue
        "ff5" :  regress given return(forward_returns) using ff5 factors and this factor, then return the time series grs pvalue
        """
        if self.stock_value is None and mode == "ff3":
            raise ValueError("You must provide stock_value when switch mode ff3.")
        elif (self.mom_signal is None) and mode == "ff3_mom":
            raise ValueError("You must provide mom_signal when switch mode ff3_mom.")
        elif (self.ag is None or self.op is None) and mode == "ff5":
            raise ValueError("You must provide ag and op when switch mode ff5.")

        if mode == "capm":
            factor_list = [s.fillna(0) for s in bundle_capm(stock_return=self.return_adj, market_cap=self.stock_size, risk_free_rate=self.risk_free_rate)]

        elif mode == "ff3":
            factor_list = [s.fillna(0) for s in bundle_ff3(stock_return=self.return_adj,
                                                           size=self.stock_size,
                                                           book_to_market= self.stock_value,
                                                           market_cap=self.stock_size,
                                                           risk_free_rate=self.risk_free_rate)]

        elif mode == "ff3_mom":
            factor_list = [s.fillna(0) for s in bundle_ff3_mom(stock_return=self.return_adj,
                                                                      size=self.stock_size,
                                                                      book_to_market=self.stock_value,
                                                                      momentum_signal=self.mom_signal,
                                                                      market_cap=self.stock_size,
                                                                      risk_free_rate=self.risk_free_rate)]

        elif mode == "ff5":

            factor_list = [s.fillna(0) for s in bundle_ff5(stock_return=self.return_adj,
                                                           size=self.stock_size,
                                                           book_to_market=self.stock_value,
                                                           profitability=self.op,
                                                           investment=self.ag,
                                                           market_cap=self.stock_size,
                                                           risk_free_rate=self.risk_free_rate)]

        factor_excess_ret = [ s.fillna(0) for s in self.factor_portfolio ]

        if mode == "single":
            concat_return = factor_excess_ret
        else:
            concat_return = factor_list + factor_excess_ret

        K = len(concat_return)
        N = self.return_adj.shape[1]

        min_win = N + K + 1
        if window <= min_win:
            raise ValueError(
                f"GRS test needs window > N + K. window={window}, N={N}, K={K}；"
            )


        concat_return_df = pd.concat(concat_return, axis=1)  # (T_full, K)
        concat_return_arr = np.stack(
            [np.tile(f.values.reshape(-1, 1), (1, N)) for f in concat_return], axis=0
        )

        rf = self.risk_free_rate.reindex(self.return_adj.index)
        excess_ret = self.return_adj.sub(rf, axis=0)
        excess_ret = excess_ret.fillna(0)
        result = RollingRegressor(x=concat_return_arr, y=excess_ret).fit(window=window)
        beta = np.array([df.fillna(0).to_numpy() for df in result.beta]).transpose(0, 2, 1)  # (K, N, T)
        alpha = np.asarray(result.alpha.fillna(0)).T  # (N, T)

        grs_pval = []
        for t in range(window - 1, beta.shape[2]):
            R_win = excess_ret.iloc[t - window + 1:t + 1, :].fillna(0).to_numpy()  # (T, N)
            F_win = concat_return_df.iloc[t - window + 1:t + 1, :].to_numpy().reshape(window,
                                                                                      -1)  # (T, K)
            B_t = beta[:, :, t]  # (K, N)
            A_t = np.tile(alpha[:, t][None, :], (window, 1))  # (T, N)

            resid = R_win - F_win @ B_t - A_t  # (T, N)

            pval = _grs(resid=resid,
                        alpha=alpha[:, t],
                        factors=F_win)
            grs_pval.append(pval)

        grs_pval_arr = np.array(grs_pval)
        index = self.return_adj.index
        grs_pval_series = pd.Series(
            data=grs_pval_arr,
            index=index[-len(grs_pval_arr):]
        ).reindex(index)
        grs_pval_series.name = "grs_pval"
        if plt :
            return plot_grs_pval(grs_pval_series)
        else:
            return grs_pval_series

    def summarize_returns(
            self,
            excess_ret: list[pd.Series],
            mode: str = "daily",
            *,
            values_are_percent: bool = False,
    ) -> pd.DataFrame:
        """
        mode ：
          - daily:  "daily", "D"
          - monthly:  "monthly", "M", "ME"
          - yearly  "annual", "yearly", "Y", "A", "YE"

        return：index=Series.name，columns=["mean excess_ret", "std"]
        """
        import pandas as pd

        freq_map = {
            "daily": "D", "D": "D",
            "monthly": "ME", "M": "ME", "ME": "ME",
            "annual": "YE", "yearly": "YE", "Y": "YE", "A": "YE", "YE": "YE",
        }
        if mode not in freq_map:
            raise ValueError(f"no such mode: {mode}")
        target = freq_map[mode]

        rows, names = [], [s.name for s in excess_ret]

        for s in excess_ret:
            if not isinstance(s.index, pd.DatetimeIndex):
                s = s.copy()
                if isinstance(s.index, pd.PeriodIndex):
                    s.index = s.index.to_timestamp(how="end")
                else:
                    s.index = pd.to_datetime(s.index)

            s = s.sort_index().dropna()
            if values_are_percent:
                s = s / 100.0
            if target == "D":
                r = s
            else:
                r = s.resample(target).apply(lambda x: (1 + x).prod() - 1)
            rows.append({
                "mean excess_ret": r.mean(),
                "std": r.std(ddof=1),
            })

        return pd.DataFrame(rows, index=names)

    def export_evaluation_table(self,
                                mode: Literal["capm","all","customize"] = "customize",
                                period: str = "daily",
                                customized_factor: list[pd.Series] | None = None,
                                cov_type: str|None = None,
                                cov_kwds: dict | None = None,
                                ) :

        """
            Generate a LaTeX-formatted evaluation table summarizing portfolio performance
            and factor regression results.

            This method computes excess returns for each factor portfolio, summarizes
            return statistics, and evaluates factor loadings under multiple models
            (CAPM, Fama–French 3-factor with momentum, and optionally a user-defined
            factor set). It outputs a combined LaTeX table containing coefficients,
            t-values, and adjusted R² values for each model, along with portfolio
            return statistics.

            Parameters
            ----------
            mode : {"daily", "monthly", ...}, default "daily"
                Frequency mode for summarizing returns. Passed to `summarize_returns`.
            customized_factor : list of pd.Series, optional
                User-specified factor return series for an additional custom factor
                regression. Only used when provided. Each series should have a datetime
                index aligned with portfolio returns.

            Returns
            -------
            str
                A LaTeX-formatted string generated by `latex_table` containing:
                - Summary statistics of excess returns
                - CAPM regression results (coefficients, t-values, adjusted R²)
                - Fama–French 3-factor with momentum regression results
                - Optional custom factor regression results (if provided)

            """

        names = [s.name for s in self.factor_portfolio]
        summarize_returns = self.summarize_returns(self.factor_portfolio, mode=period)
        summarize_returns.index = names



        if mode == "customize":
            if customized_factor is None:
                raise ValueError(f"customize mode requires customize_factors")
            mkt = None
            mkt_r2_adj = None
            ff4 = None
            ff4_r2_adj = None
            customized_factor_df, customized_factor_stats_df, customized_factor_r2_adj = self.evaluate_by_other_factors(mode="customize",
                                                                                                                        factor2=customized_factor,
                                                                                                                        cov_type=cov_type,
                                                                                                                        cov_kwds=cov_kwds,
                                                                                                                        )
            customized_factor = stitch_coeff_tvalue(customized_factor_df, customized_factor_stats_df)
        elif mode == "all":
            if (self.return_adj is None) or (self.stock_size is None) or (self.mom_signal is None):
                raise ValueError(f"capm requires return_adj, stock_size, mom_signal")
            mkt_df, mkt_stats_df, mkt_r2_adj = self.evaluate_by_other_factors(mode="capm",cov_type=cov_type, cov_kwds=cov_kwds)
            ff4_df, ff4_stats_df, ff4_r2_adj = self.evaluate_by_other_factors(mode="ff3_mom",cov_type=cov_type, cov_kwds=cov_kwds)
            mkt = stitch_coeff_tvalue(mkt_df, mkt_stats_df)
            ff4 = stitch_coeff_tvalue(ff4_df, ff4_stats_df)
            if customized_factor is None:
                customized_factor = None
                customized_factor_r2_adj = None
            else:
                customized_factor_df, customized_factor_stats_df, customized_factor_r2_adj = self.evaluate_by_other_factors(
                    mode="customize", factor2=customized_factor,cov_type=cov_type, cov_kwds=cov_kwds)
                customized_factor = stitch_coeff_tvalue(customized_factor_df, customized_factor_stats_df)
        elif mode == "capm":
            if (self.return_adj is None) or (self.stock_size is None):
                raise ValueError(f"capm requires return_adj and stock_size")
            mkt_df, mkt_stats_df, mkt_r2_adj = self.evaluate_by_other_factors(mode="capm",cov_type=cov_type, cov_kwds=cov_kwds)
            mkt = stitch_coeff_tvalue(mkt_df, mkt_stats_df)
            ff4 = None
            ff4_r2_adj = None
            if customized_factor is None:
                customized_factor = None
                customized_factor_r2_adj = None
            else:
                customized_factor_df, customized_factor_stats_df, customized_factor_r2_adj = self.evaluate_by_other_factors(
                    mode="customize", factor2=customized_factor,cov_type=cov_type, cov_kwds=cov_kwds)
                customized_factor = stitch_coeff_tvalue(customized_factor_df, customized_factor_stats_df)
        else:
            raise ValueError(f"no such mode: {mode}")

        return latex_table(df1=mkt,df1_r2=mkt_r2_adj,
                           df2=ff4,df2_r2=ff4_r2_adj,
                           customize_factor=customized_factor,customize_r2=customized_factor_r2_adj,
                           excess_df=summarize_returns)


def stitch_coeff_tvalue(df_coeff: pd.DataFrame, df_tval: pd.DataFrame) -> pd.DataFrame:
    if not df_coeff.index.equals(df_tval.index):
        raise ValueError("Index mismatch between coeff and tvalue DataFrames")
    if not df_coeff.columns.equals(df_tval.columns):
        raise ValueError("Column mismatch between coeff and tvalue DataFrames")

    out = pd.concat({'coeff': df_coeff, 'tvalue': df_tval}, axis=1) \
            .swaplevel(0, 1, axis=1)

    cols = df_coeff.columns
    out = out.reindex(columns=pd.MultiIndex.from_product([cols, ['coeff', 'tvalue']]))
    return out
