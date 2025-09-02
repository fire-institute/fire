# -*- coding: utf-8 -*-
# @Created : 2025/3/26 17:01
# @Author  : Liao Renjie
# @Email   : liao.renjie@techfin.ai
# @File    : least_square.py
# @Software: PyCharm

import textwrap
import typing

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

__all__ = ["least_square", "RollingRegressor", "rolling_regression", "table_regression"]

NotProvided = object()


class RegressionResult:
    """
    Encapsulate the results from `least_square`.

    Parameters
    ----------
    sm_result: sm.regression.linear_model.RegressionResults
        The regression results object from the statsmodels library.
    fit_intercept: bool
        Whether to fit an intercept term.
    univariate: bool
        Whether it is a univariate regression.

    """

    def __init__(self, sm_result: sm.regression.linear_model.RegressionResults, fit_intercept: bool, univariate: bool):
        self.sm_result = sm_result
        self.fit_intercept = fit_intercept
        self.univariate = univariate

    @property
    def alpha(self):
        """float or None"""
        if self.fit_intercept:
            return self.sm_result.params[0]
        else:
            return None

    @property
    def beta(self):
        """1D array if multivariate or float if univariate"""
        if self.univariate:
            return self.sm_result.params[-1]
        else:
            # multivariate
            if self.fit_intercept:
                return self.sm_result.params[1:]
            else:
                return self.sm_result.params

    @property
    def t_alpha(self):
        """float or None —> t‑stat of intercept α."""
        if self.fit_intercept:
            return self.sm_result.tvalues[0]
        else:
            return None

    @property
    def tvalue(self):
        """t‑statistics corresponding to the beta coefficients (intercept excluded if `fit_intercept`)."""
        if self.univariate:
            return self.sm_result.tvalues[-1]
        else:
            if self.fit_intercept:
                return self.sm_result.tvalues[1:]
            else:
                return self.sm_result.tvalues

    @property
    def r2(self):
        """
        Return the coefficient of determination R² of the regression.

        Returns
        -------
        float
            The R² value.
        """
        return self.sm_result.rsquared

    @property
    def r2_adj(self):
        """
        Return the adjusted coefficient of determination R² of the regression.

        Returns
        -------
        float
            The adjusted R² value.
        """
        return self.sm_result.rsquared_adj

    @property
    def residuals(self):
        """
        Return the residuals of the regression.

        Returns
        -------
        array
            The array of residuals.
        """
        return self.sm_result.resid


class BatchRegressionResult:
    """
    Encapsulate the results of batch regression.

    Parameters
    ----------
    beta
        The regression coefficients.
    alpha: optional
        The intercept term, default is None.
    tvalue: optional
        The t‑statistics corresponding to `beta`, default is None.
    r2: optional
        The coefficient of determination R², default is None.
    r2_adj: optional
        The adjusted coefficient of determination R², default is None.
    residuals: optional
        The residuals, default is None.
    """

    def __init__(
        self,
        beta,
        *,
        alpha=None,
        tvalue=None,
        alpha_t=None,
        r2=None,
        r2_adj=None,
        residuals=None,
    ):
        # NOTE: public names will be displayed in __repr__
        self.alpha = alpha
        self.beta = beta
        self.tvalue = tvalue
        self.alpha_t = alpha_t
        self.r2 = r2
        self.r2_adj = r2_adj
        self.residuals = residuals

    def __repr__(self):
        content = {a: getattr(self, a) for a in dir(self) if not a.startswith("_")}
        content = ",\n".join(
            [f" {k}:\n{textwrap.indent(repr(v), prefix='    ')}" for k, v in content.items() if v is not None]
        )
        return f"{self.__class__.__name__}(\n{content}\n)"


def _regression(
    x: pd.DataFrame | pd.Series,
    y: pd.Series,
    w: pd.Series = None,
    fit_intercept: bool = True,
    cov_type: str | None = None,
    cov_kwds: dict | None = None,
) -> sm.regression.linear_model.RegressionResults:
    """
    Perform a linear regression using either OLS or WLS.

    Parameters
    ----------
    x: pd.DataFrame | pd.Series
        The independent variable(s).
    y: pd.Series
        The dependent variable.
    w: pd.Series, optional
        The weights for WLS, default is None.
    fit_intercept: bool, optional
        Whether to fit an intercept term, default is True.
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
    sm.regression.linear_model.RegressionResults
        The regression results.
    """

    # if x contains nan, fill nan with 0
    # TODO: fill nan with 0 is not a good idea, we should use the mean of the column to fill nan
    x = np.nan_to_num(x, nan=0)
    y = np.nan_to_num(y, nan=0)

    if fit_intercept:
        x = sm.add_constant(x)
    if w is None:
        model = sm.OLS(y, x)
    else:
        # TODO: fill nan with 0 is not a good idea, we should use the mean of the column to fill nan
        y = np.nan_to_num(y, nan=0)
        model = sm.WLS(y, x, weights=w)

    if cov_type is None:
        return model.fit()
    else:
        return model.fit(cov_type=cov_type, cov_kwds=cov_kwds or {})


def least_square(
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray,
    y: pd.Series | np.ndarray,
    w: pd.Series | np.ndarray | None = None,
    fit_intercept: bool = True,
) -> RegressionResult:
    """
    A simple wrapper around sm.OLS or sm.WLS.

    Parameters
    ----------
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The independent variable(s). If one-dimensional, the regression is considered univariate, otherwise is
        considered multivariate. This affects the format of returned beta.
    y: pd.Series | np.ndarray
        The dependent variable.
    w: pd.Series | np.ndarray | None, optional
        The weights for WLS, default is None.
    fit_intercept: bool, optional
        Whether to fit an intercept term, default is True.

    Returns
    -------
    RegressionResult
        The regression result object.
    """
    if isinstance(x, (tuple, list)):
        x = pd.concat(x, axis=1)

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be 1d or 2d array")

    univariate = x.shape[1] == 1

    result = _regression(x, y, w=w, fit_intercept=fit_intercept)
    return RegressionResult(result, fit_intercept=fit_intercept, univariate=univariate)


@delayed
def calculate_window(x_wind, y_wind, w_wind, m1, m2, m3, m, fit_intercept, univariate, cov_type, cov_kwds):
    alphas: list[float] = []
    betas: list[float | np.ndarray | None] = []
    tvals: list[float | np.ndarray | None] = []
    talphas: list[float] = []
    r2s: list[float] = []
    r2adjs: list[float] = []
    resmats = []

    win_len = x_wind.shape[1]

    for j in range(m):
        x_j = x_wind[:, :, min(j, m1 - 1)].T
        y_j = y_wind[:, min(j, m2 - 1)]
        w_j = None if w_wind is None else w_wind[:, min(j, m3 - 1)]
        # if any x is all nan, skip regression
        if np.isnan(x_j).all(axis=0).any() or np.isnan(y_j).all():
            alpha = np.nan
            beta = None
            tval = None
            ta = np.nan
            r2_val = np.nan
            r2_adj_val = np.nan
            resid_arr = np.full(win_len, np.nan)
        else:
            res = RegressionResult(
                # fit_intercept is always False, because we've padded X in __init__
                _regression(x_j, y_j, w_j, fit_intercept=False, cov_type=cov_type, cov_kwds=cov_kwds),
                fit_intercept=fit_intercept,
                univariate=univariate,
            )
            alpha = res.alpha
            beta = res.beta
            tval = res.tvalue
            ta = res.t_alpha
            r2_val = res.r2
            r2_adj_val = res.r2_adj
            resid_arr = res.residuals

        alphas.append(alpha)
        betas.append(beta)
        tvals.append(tval)
        talphas.append(ta)
        r2s.append(r2_val)
        r2adjs.append(r2_adj_val)
        resmats.append(resid_arr)

    residual_mat = np.column_stack(resmats) #list[1D] → 2D (window × m)

    return alphas, betas, tvals, talphas, r2s, r2adjs, residual_mat


class RollingRegressor:
    """
    Perform rolling regression.

    Parameters
    ----------
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The independent variable(s).
    y: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The dependent variable.
    w: optional
        The weights for WLS, default is None.
    mode: typing.Literal["single", "multi"], optional
        The mode of regression, default is None.
    fit_intercept: bool, optional
        Whether to fit an intercept term, default is True.
    """

    def __init__(
        self,
        x,
        y,
        w=None,
        *,
        mode: typing.Literal["single", "multi"] = None,
        fit_intercept: bool = True,
    ):
        # We generally don't check the alignment of the inputs. It's the user's obligation to make sure the inputs are
        # compatible in turns of shape and align with each other.
        self._keys = {}
        self._index = {}
        self._columns = {}
        self.x = self._parse_data(x, "x")
        self.y = self._parse_data(y, "y")
        self.w = self._parse_data(w, "w", allow_none=True)

        # "multi": x is 3d array
        # "single": x is 2d array
        if isinstance(self.x, np.ndarray):
            if self.x.ndim == 2:
                self.inferred_mode = "single"
                self.x = self.x.reshape(1, *self.x.shape)
            elif self.x.ndim == 3:
                self.inferred_mode = "multi"
            else:
                raise ValueError("x must be 2d or 3d array")
        else:
            raise ValueError("parsed x should be array")

            # now x is 3d array: key-index-columns
        if fit_intercept:
            self.x = np.concatenate([np.ones((1, *self.x.shape[1:])), self.x])

        if mode is not None and mode != self.inferred_mode:
            raise ValueError(f"inferred mode ({self.inferred_mode}) is not equal to the specified mode ({mode})")

        self.keys = None if not self._keys else next(iter(self._keys.values()))
        self.index = None if not self._index else next(iter(self._index.values()))

        if not self._columns:
            self.columns = None
        else:
            len_col = list(map(len, self._columns.values()))
            max_len_loc = len_col.index(max(len_col))
            self.columns = list(self._columns.values())[max_len_loc]

        self.fit_intercept = fit_intercept

    @property
    def is_univariate(self):
        """
        Check if the regression is univariate.

        Returns
        -------
        bool
            True if univariate, False otherwise.
        """
        if self.inferred_mode == "single":
            return True
        else:
            assert self.inferred_mode == "multi"
            return False

    def _parse_data(self, a, data_name: typing.Literal["x", "y", "w"], allow_none=False):
        """
        Parse the input data.

        Parameters
        ----------
        a
            The input data.
        data_name: typing.Literal["x", "y", "w"]
            The name of the data.
        allow_none: bool, optional
            Whether to allow None as input, default is False.

        Returns
        -------
        np.ndarray
            The parsed data.
        """
        if a is None:
            if allow_none:
                return
            else:
                raise ValueError(f"input {data_name} cannot be None")
        if isinstance(a, pd.Series):
            a = a.to_frame()
        if isinstance(a, pd.DataFrame):
            self._index[data_name] = a.index
            self._columns[data_name] = a.columns
            return a.values
        elif isinstance(a, np.ndarray):
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.ndim not in (2, 3):
                raise ValueError(f"input {data_name} should be 2-d or 3-d if it's array")
            return a
        else:
            if data_name in ("x", "w"):
                if isinstance(a, dict):
                    self._keys[data_name] = list(a.keys())
                    a = list(a.values())

                if isinstance(a, (list, tuple)):
                    if len(set([i.shape for i in a])) != 1:
                        raise ValueError(f"input {data_name} should have same shape")
                    if not all(i.ndim == 2 for i in a):
                        raise ValueError(f"input contents of {data_name} should be 2-d, if it's list")
                    if isinstance(a[0], pd.DataFrame):
                        self._index[data_name] = a[0].index
                        self._columns[data_name] = a[0].columns
                    a = np.array(a)
                    return a
                else:
                    raise TypeError(f"input {data_name} should be array-like or list")

            else:
                raise ValueError(f"input {data_name}'s type not supported")

    @classmethod
    def _transpose_or_none(cls, _x):
        """
        Transpose the array if it is not None.

        Parameters
        ----------
        _x
            The input array.

        Returns
        -------
        np.ndarray or None
            The transposed array or None.
        """
        # the last 2 axes are always time x stocks
        if _x is not None:
            return np.swapaxes(_x, -1, -2)

    def fit(
        self,
        window: int | None = None,
        axis=0,
        cov_type: str | None = None,
        cov_kwds: dict | None = None,
        n_jobs: int = 4,
        verbose: int = 0,
    ):
        """
        Fit the rolling regression model.

        Parameters
        ----------
        window: int | None, optional
            The window size for rolling regression, default is None. If None, window = len(data)
        axis: int, optional
            The axis along which to perform the regression, default is 0.
        cov_type: str | None, optional
            The covariance estimator, default is None.
            - If None: use the default homoskedastic standard errors.
            - If "HAC": Newey–West heteroskedasticity-and-autocorrelation robust SE.
            - Other options supported by statsmodels (e.g. "HC0", "HC1", …).
        cov_kwds: dict | None, optional
            The keyword arguments for the covariance estimator, default is None.
            For Newey–West, you’d typically pass `{"maxlags": L}` to control lag length.
        n_jobs: int
            num of parallel workers, passed to Parallel
        verbose: int
            verbosity of progress, passed to Parallel

        Returns
        -------
        BatchRegressionResult
            The batch regression result object.
        """
        x = self.x
        y = self.y
        w = self.w

        keys = self.keys
        index = self.index
        columns = self.columns

        fit_intercept = self.fit_intercept
        univariate = self.inferred_mode == "single"
        transpose = axis != 0

        if transpose:
            x = self._transpose_or_none(x)
            y = self._transpose_or_none(y)
            w = self._transpose_or_none(w)

        # generic shape compat
        k, n1, m1 = x.shape
        n2, m2 = y.shape
        if m1 != m2 and min(m1, m2) != 1:
            raise ValueError(f"incompatible x, y shapes: {x.shape} vs {y.shape}")
        if n1 != n2:
            raise ValueError(f"x, y should have same length")

        n = n1
        m = max(m1, m2)
        m3 = 1

        if w is not None:
            n3, m3 = w.shape
            if m3 > 1 and m3 != m:
                raise ValueError(f"incompatible x, y, w shapes: {x.shape} vs {y.shape} vs {w.shape}")
            if n3 != n:
                raise ValueError(f"x, w should have same length")

        # window not specified, use total length as window
        # in this case, result should also be pruned
        is_table = window is None
        if is_table:
            window = n

        alpha = np.full((n, m), np.nan)
        alpha_t = np.full((n, m), np.nan)
        beta = np.full((k - fit_intercept, n, m), np.nan)
        tvalue = np.full((k - fit_intercept, n, m), np.nan)
        r2 = np.full((n, m), np.nan)  # >>> 新增 R²：矩阵
        r2_adj = np.full((n, m), np.nan)
        use_dict = (window is not None and axis == 0)  
        residuals_dict = {} if use_dict else None
        residual_full = None

        result_gen = Parallel(n_jobs=n_jobs, verbose=verbose, return_as="generator")(
            calculate_window(
                x_wind=x[:, i: i + window],
                y_wind=y[i: i + window],
                w_wind=None if w is None else w[i: i + window],
                m1=m1,
                m2=m2,
                m3=m3,
                m=m,
                fit_intercept=fit_intercept,
                univariate=univariate,
                cov_type=cov_type,
                cov_kwds=cov_kwds,
            )
            for i in range(n - window + 1)
        )
        for i, (alphas, betas, tvals, talphas, r2s, r2adjs, resmat) in enumerate(result_gen):
            alpha[i + window - 1] = alphas
            alpha_t[i + window - 1] = talphas
            r2[i + window - 1] = r2s 
            r2_adj[i + window - 1] = r2adjs
            for j, (_beta, _t) in enumerate(zip(betas, tvals)):
                if _beta is not None:
                    beta[:, i + window - 1, j] = _beta
                    tvalue[:, i + window - 1, j] = _t

            if use_dict:
                end_key = self.index[i + window - 1] if self.index is not None else i + window - 1
                idx_window = self.index[i: i + window] if self.index is not None else range(i, i + window)
                col_labels = self.columns
                residuals_dict[end_key] = pd.DataFrame(resmat, index=idx_window, columns=col_labels)
            else:
                residual_full = resmat

        # squeeze if table
        if is_table:
            # columns
            if transpose:  # axis = 1（横截面回归）：取按“时间”的序列
                alpha = alpha[-1]  # 长度 = 天数（m）
                alpha_t = alpha_t[-1]
            else:  # axis = 0：取“最后一个时间点”的截面（按股票）
                alpha = alpha[-1]  # 长度 = 股票数（m）
                alpha_t = alpha_t[-1]
            r2 = r2[-1] 
            r2_adj = r2_adj[-1]
            # keys x columns
            beta = beta[:, -1]
            tvalue = tvalue[:, -1]
        # maybe transpose back
        if transpose:
            beta = self._transpose_or_none(beta)
            tvalue = self._transpose_or_none(tvalue)
        # wrap dataframe if possible
        if is_table:
            if transpose:  # axis = 1：返回时序 Series（索引=时间）
                alpha = pd.Series(alpha, index=index, name="alpha")
                alpha_t = pd.Series(alpha_t, index=index, name="alpha_t")
            else:  # axis = 0：返回 1×N DataFrame（最后一个时间点 × 股票）
                last_idx = index[-1] if index is not None else (n - 1)
                alpha = pd.DataFrame([alpha], index=[last_idx], columns=columns)
                alpha_t = pd.DataFrame([alpha_t], index=[last_idx], columns=columns)
            r2 = pd.Series(r2, index=index if transpose else columns, name="r2") 
            r2_adj = pd.Series(r2_adj, index=index if transpose else columns, name="r2_adj")
            if transpose:
                # axis = 1
                beta = pd.DataFrame(beta, index=index, columns=keys)
                tvalue = pd.DataFrame(tvalue, index=index, columns=keys)
            else:
                beta = pd.DataFrame(beta, index=keys, columns=columns)
                tvalue = pd.DataFrame(tvalue, index=keys, columns=columns)
            if self.is_univariate:
                beta = beta.squeeze(axis=axis)
                tvalue = tvalue.squeeze(axis=axis)
        else:
            alpha = pd.DataFrame(alpha, index=index, columns=columns)
            alpha_t = pd.DataFrame(alpha_t, index=index, columns=columns)
            r2 = pd.DataFrame(r2, index=index, columns=columns)  
            r2_adj = pd.DataFrame(r2_adj, index=index, columns=columns)
            if self.is_univariate:
                beta = pd.DataFrame(np.squeeze(beta, axis=0), index=index, columns=columns)
                tvalue = pd.DataFrame(np.squeeze(tvalue, axis=0), index=index, columns=columns)
            else:
                beta = [pd.DataFrame(beta[i], index=index, columns=columns) for i in range(k - fit_intercept)]
                tvalue = [pd.DataFrame(tvalue[i], index=index, columns=columns) for i in range(k - fit_intercept)]
                if keys is not None:
                    for _key, _beta in zip(keys, beta):
                        _beta.name = _key
                    for _key, _tv in zip(keys, tvalue):
                        _tv.name = _key

        if not use_dict:
            if transpose:  
                residual_full = residual_full.T
            residuals_out = pd.DataFrame(residual_full, index=self.index, columns=self.columns)
        else:
            residuals_out = residuals_dict

        return BatchRegressionResult(beta, alpha=alpha, tvalue=tvalue, alpha_t=alpha_t, r2=r2, r2_adj=r2_adj, residuals=residuals_out)


def rolling_regression(x, y, window, w=None, cov_type: str | None = None, *, fit_intercept=True):
    """
    Perform rolling regression.

    Parameters
    ----------
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The independent variable(s).
    y: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The dependent variable.
    window: int
        The window size for rolling regression.
    w: optional
        The weights for WLS, default is None.
    fit_intercept: bool, optional
        Whether to fit an intercept term, default is True.

    Returns
    -------
    BatchRegressionResult
        The batch regression result object.
    """
    return RollingRegressor(x, y, w, fit_intercept=fit_intercept).fit(window, cov_type= cov_type)


def table_regression(x, y, w=None, *, fit_intercept=True, axis=1):
    """
    Perform table regression (apply regression column-wise or row-wise)

    Parameters
    ----------
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The independent variable(s).
    y: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray
        The dependent variable.
    w: optional
        The weights for WLS, default is None.
    fit_intercept: bool, optional
        Whether to fit an intercept term, default is True.
    axis: int, optional
        The axis along which to perform the regression, default is 1.

    Returns
    -------
    BatchRegressionResult
        The batch regression result object.
    """
    return RollingRegressor(x, y, w, fit_intercept=fit_intercept).fit(None, axis=axis)
