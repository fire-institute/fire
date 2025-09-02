# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

# TODO: Move some common algorithms to fire/core/algorithm/

import typing

import numpy as np
import pandas as pd

__all__ = [
    "compute_forward_returns",
    "compute_ic",
    "factor_to_quantile",
    "factor_to_quantile_dependent_double_sort",
    "compute_quantile_returns",
    "_compute_weighted_quantile_df",
    "_compute_quantile_df",
]

PeriodType = typing.NewType("PeriodType", int)
ForwardReturns = typing.NewType("ForwardReturns", dict[PeriodType, pd.DataFrame])
IC = typing.NewType("IC", pd.DataFrame)
QuantileReturns = typing.NewType("QuantileReturns", dict[PeriodType, pd.DataFrame])


def compute_forward_returns(price: pd.DataFrame, periods: list[PeriodType]) -> ForwardReturns:
    '''
    Compute forward returns over specified holding periods.

    Parameters
    ----------
    price : pd.DataFrame
        Asset adjusted price time series with shape (Time × Stock).
        Each column represents the adjusted price of an asset indexed by date.
        
    periods : list[PeriodType]
        List of forward periods to compute returns for.

    Returns
    -------
    ForwardReturns
        A wrapper around a dictionary of DataFrames:
        - Keys are the holding periods.
        - Values are DataFrames of forward returns with same shape as price (Time × Stock),
          where each value is the forward return from time t to t + period for each asset.

    '''
    if np.any(price.values <= 0):
        raise ValueError('Price is adjusted price, which must be greater than 0.')
    
    forward_returns_dict = {}

    returns: pd.DataFrame = np.log(price).shift(-1) - np.log(price)

    for period in sorted(periods):
        if period == 1:
            forward_returns_dict[period] = returns
            continue

        log_period_returns = returns.rolling(period).sum().shift(1 - period)
        period_returns: pd.DataFrame = np.exp(log_period_returns) - 1
        forward_returns_dict[period] = period_returns
    return ForwardReturns(forward_returns_dict)


def _compute_ic_df_df(
    a: pd.DataFrame, 
    b: pd.DataFrame, 
    method: typing.Literal["pearson", "kendall", "spearman"]
) -> pd.Series:
    '''
     Compute the row-wise correlation (Information Coefficient, IC) between two DataFrames.

    Each row is treated as a cross-sectional slice (e.g., a specific date),
    and the correlation is calculated between the corresponding rows of `a` and `b`.

    Note:
    -----
    `a` and `b` should have the same index (e.g., dates) and columns (e.g., assets).
    If they differ, the correlation will be computed based on the intersection 
    of their indices and columns for each row.

    Parameters
    ----------
    a : pd.DataFrame
        First DataFrame with rows representing dates and columns representing assets.
    b : pd.DataFrame
        Second DataFrame with rows representing dates and columns representing assets.
    method : {"pearson", "kendall", "spearman"}
        Correlation method to use. "pearson" is the default

    Returns
    -------
    pd.Series
        A Series of correlation values (IC), indexed by the row labels (typically dates).
    '''

    return a.corrwith(b, axis = 1, method = method)


def compute_ic(
    factor: pd.DataFrame, 
    forward_returns: ForwardReturns, 
    method: typing.Literal["pearson", "kendall", "spearman"]
) -> IC:
    """
    Compute IC (Information Coefficient) for the factor and forward returns, which is the correlation between the
    factor and the forward returns.

    Parameters
    ----------
    factor: pd.DataFrame
    forward_returns: ForwardReturns
    method: str
        default "pearson"

    Returns
    -------
    IC
        a dataframe of IC values for each period in columns.

    """
    factor = factor[np.isfinite(factor)]
    return IC(
        pd.DataFrame(
            {
                period: _compute_ic_df_df(factor, period_returns, method=method)
                for period, period_returns in forward_returns.items()
            }
        )
    )

def summarise_ic(ic_data: IC) -> pd.DataFrame:
    '''
    Generate a summary table of key statistics for the Information Coefficient (IC) time series.

    This function computes descriptive metrics for each holding period (i.e., each column in `ic_data`), 
    including the mean, standard deviation, Information Ratio (IR), and the proportion of IC values 
    exceeding common significance thresholds.

    Metrics included:
    - mean: Average IC
    - std: Standard deviation of IC
    - ir: Information Ratio (mean / std)
    - > 0: Proportion of IC values greater than 0
    - < 0: Proportion of IC values less than 0
    - > 3% / < -3%: Proportion of IC values greater than 3% or less than -3%
    - > 5% / < -5%: Proportion of IC values greater than 5% or less than -5%

    Parameters
    ----------
    ic_data : IC
        A DataFrame where each column represents IC values for a specific holding period 
        and each row corresponds to a time point (e.g., daily IC).

    Returns
    -------
    pd.DataFrame
        A summary table with statistics for each holding period.
    '''

    summary_table = pd.DataFrame(
        np.nan,
        index = ["mean", "std", "ir", "> 0", "< 0", "> 3%", "< -3%", "> 5%", "< -5%"],
        columns = ic_data.columns,
    )

    ic_mean = ic_data.mean()
    ic_std = ic_data.std()
    ir = ic_mean / ic_std

    summary_table.loc["mean"] = ic_mean.values
    summary_table.loc["std"] = ic_std.values
    summary_table.loc["ir"] = ir.values
    summary_table.loc["mean"] = ic_mean.values
    summary_table.loc["std"] = ic_std.values
    summary_table.loc["ir"] = ir.values
    summary_table.loc["> 0"] = ((ic_data > 0).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< 0"] = ((ic_data < 0).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["> 3%"] = ((ic_data > 0.03).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< -3%"] = ((ic_data < -0.03).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["> 5%"] = ((ic_data > 0.05).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< -5%"] = ((ic_data < -0.05).sum() / np.isfinite(ic_data).sum()).values

    return summary_table

def generate_latex_code(plot_path: str, summary_table: pd.DataFrame) -> str:
    '''
    Generate complete LaTeX codes as a string, embedding a plot image and a summary table.
    
    Parameters
    ----------
    plot_path : str
        The file path to the image to include in the LaTeX codes.
    summary_table : pd.DataFrame
        A pandas DataFrame containing the summary statistics that will be rendered as a LaTeX table.

    Returns
    -------
    str
        Full LaTeX codes as a single string, ready to be written to a .tex file.
    
    '''
    latex_code = [
    r'\documentclass[a4paper]{article}',
    r'\usepackage[margin=2cm]{geometry}',
    r'\usepackage{graphicx}',
    r'\usepackage{booktabs}',
    r'\usepackage{float}',
    r'\begin{document}',
    '',
    r'\section*{Factor Analysis Result}',
    r'\subsection*{IC Plot}',
    f'\includegraphics[width=1\\textwidth]{{{plot_path}}}',
    r'\subsection*{IC Summary Table}',
    f"{summary_table.to_latex(float_format = '%.4f', escape = True)}",
    r'\end{document}'
    ]
    latex_code = '\n'.join(latex_code).replace('_', '\_')

    return latex_code

def factor_to_quantile(factor: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """
    Convert factor to quantile row-wise. The result will always have quantile values ranging from `quantiles` down
    to 1 continuously (if only 1 group, it'll be `quantiles`).

    Parameters
    ----------
    factor: pd.DataFrame
    quantiles: int
        default 5

    Returns
    -------
    pd.DataFrame
        a dataframe of quantile values.

    """
    quantile_values = np.arange(1, quantiles + 1)

    def _row_to_quantile(row):
        finite = np.isfinite(row)
        if finite.any():
            tmp: pd.Series = pd.qcut(row[finite], quantiles, labels=False, duplicates="drop")
            # rearrange values from `q` to 1
            # this makes sure that the quantile values are generally continuous,
            # and we always have a group of long portfolio of `q`
            old_values = tmp.unique()
            old_values.sort()
            new_values = quantile_values[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                tmp.replace(old_values, new_values, inplace=True)
            row = row.copy()
            row[finite] = tmp
            return row
        else:
            return row

    return factor.apply(_row_to_quantile, axis=1)

def factor_to_quantile_dependent_double_sort(primary_factor: pd.DataFrame, secondary_factor: pd.DataFrame, quantiles: typing.Tuple[int, int]):
    """
    Perform dependent double sorting on two factors.

    Parameters:
    ------------
    primary_factor : pd.DataFrame
        The primary factor used for initial sorting.
    secondary_factor : pd.DataFrame
        The secondary factor used for sorting within each group defined by the primary factor.
    quantiles : tuple of int
       A tuple containing the number of quantiles for the primary and secondary factors respectively.
    
    Returns:
    --------
    quantile_sorts : pd.DataFrame
       A DataFrame where each entry represents the quantile assignment for the secondary factor within the group defined by the primary factor.
    
    TODO: numba jit acceleration
    """
    quantile_values_p = np.arange(1, quantiles[0] + 1)
    quantile_values_s = np.arange(1, quantiles[1] + 1)

    def _row_to_quantile(row_p, row_s):
        finite_p = np.isfinite(row_p)
        finite_s = np.isfinite(row_s)

        if finite_p.any() or finite_s.any():
            # Sort by primary factor first
            temp_p : pd.Series = pd.qcut(row_p[finite_p], quantiles[0], labels=False, duplicates='drop') 
            old_values = temp_p.unique()
            old_values.sort()
            new_values = quantile_values_p[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                temp_p.replace(old_values, new_values, inplace=True)

            # Sort by secondary factor within each primary quantile
            temp_s = pd.Series(np.zeros_like(row_p), index=row_p.index, dtype=int)
            temp_s[~finite_p | ~finite_s] = np.nan

            for q in quantile_values_p:
                mask = temp_p == q
                if mask.any():
                    # nan + nan, nan + int -> nan, int + nan -> nan, int + int -> int
                    temp_s[mask] = pd.qcut(row_s[finite_s & mask], quantiles[1], labels=False, duplicates='drop')
                else:
                    temp_s[mask] = np.nan
            
            old_values = temp_s.unique()
            old_values.sort()
            new_values = quantile_values_s[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                temp_s.replace(old_values, new_values, inplace=True)
            
            return temp_p.astype(str) + "_" + temp_s.astype(str)
        else:
            return pd.Series(index=row_p.index, dtype=str)

    result = pd.DataFrame(index=primary_factor.index, columns=primary_factor.columns)
    # apply the function to each row both of the factors
    for (i, row_p), (_, row_s) in zip(primary_factor.iterrows(), secondary_factor.iterrows()):
        result.loc[i] = _row_to_quantile(row_p, row_s)

    return result

def _compute_quantile_df(qt: pd.DataFrame, fr: pd.DataFrame, reindex=True, quantiles: int = 5):
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row) in zip(fr.iterrows(), qt.iterrows()):
        result[dt] = fr_row.groupby(qt_row).mean()
    result = pd.DataFrame(result).T
    if reindex:
        return result.reindex(columns=np.arange(1, quantiles + 1), copy=False)
    return result

def _compute_weighted_quantile_df(qt: pd.DataFrame, fr: pd.DataFrame, wt: pd.DataFrame, reindex= True, quantiles: int = 5):
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row), (_, wt_row) in zip(fr.iterrows(), qt.iterrows(), wt.iterrows()):
        _wt_row = wt_row.groupby(qt_row).transform(lambda x: x / x.sum())
        result[dt] = (fr_row * _wt_row).groupby(qt_row).sum()
    result = pd.DataFrame(result).T
    if reindex:
        return result.reindex(columns=np.arange(1, quantiles + 1), copy=False)
    return result

def compute_quantile_returns(
    factor: pd.DataFrame, forward_returns: ForwardReturns, quantiles: int = 5
) -> QuantileReturns:
    """
    Compute quantile returns. Factor will be converted to quantiles using `factor_to_quantile`. Then, for each period
    in forward_returns, the period returns will be grouped row-wise by quantiles and averaged.

    Parameters
    ----------
    factor: pd.DataFrame
    forward_returns: ForwardReturns
    quantiles: int
        default 5

    Returns
    -------
    QuantileReturns
        a dictionary of period returns for each quantile. The quantile returns are dataframe with index as date and
        columns as quantiles.

    """
    factor_as_quantile = factor_to_quantile(factor, quantiles=quantiles)
    return QuantileReturns(
        {
            period: _compute_quantile_df(factor_as_quantile, period_returns, quantiles=quantiles)
            for period, period_returns in forward_returns.items()
        }
    )

def double_sort_returns_independent(
    size: pd.DataFrame,
    x: pd.DataFrame,
    return_adj: pd.DataFrame,
    value_weighted: bool = True,
    market_cap: typing.Optional[pd.DataFrame] = None,
    size_break: float = 0.5,
    x_breaks: typing.Tuple[float, float] = (0.3, 0.7),
    lag_weights : bool = True,
) -> pd.DataFrame:
    """
    用独立 2×3（Size=50%、X=30/70，日频断点）计算六个组合的日收益（列: '1_1'..'2_3'）。
    """

    def double_sort_labels_independent(
            size: pd.DataFrame,
            x: pd.DataFrame,
            size_break: float = 0.5,
            x_breaks: typing.Tuple[float, float] = (0.3, 0.7),
    ) -> pd.DataFrame:
        """
        Academic independent 2×3 labels with DAILY breakpoints.
          - Size: 1=Small (rank_pct <= 0.5), 2=Big (rank_pct > 0.5)
          - X   : 1=Low (rank_pct <= 0.3), 2=Mid (0.3 < rank_pct <= 0.7), 3=High (rank_pct > 0.7)
        返回: (Time×Stock) 的字符串标签 DataFrame，取值如 '1_1'..'2_3'。
        """
        assert size.index.equals(x.index) and size.columns.equals(x.columns), "size/x must align"

        # 用横截面“秩百分位”做断点，稳健处理并列值；索引与列均保持原样
        size_rank = size.rank(axis=1, method="first", pct=True)
        x_rank = x.rank(axis=1, method="first", pct=True)

        # Size 分两组
        g1 = pd.DataFrame(np.nan, index=size.index, columns=size.columns)
        g1[size_rank <= size_break] = 1
        g1[size_rank > size_break] = 2

        # X 分三组
        lo, hi = x_breaks
        g2 = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        g2[x_rank <= lo] = 1
        g2[(x_rank > lo) & (x_rank <= hi)] = 2
        g2[x_rank > hi] = 3

        # 组合成 'i_j' 标签
        labels = pd.DataFrame(index=size.index, columns=size.columns, dtype=object)
        mask = g1.notna() & g2.notna()

        g1s = g1.where(mask).astype("Int64").astype(str)  # 用可空整数，允许 NA
        g2s = g2.where(mask).astype("Int64").astype(str)

        labels[mask] = (g1s + "_" + g2s)[mask]
        return labels

    assert return_adj.index.equals(size.index) and return_adj.columns.equals(size.columns), "return_adj/size mismatch"
    assert size.index.equals(x.index) and size.columns.equals(x.columns), "size/x mismatch"

    labels = double_sort_labels_independent(size, x, size_break, x_breaks)

    if value_weighted:
        assert market_cap is not None, "market_cap required for value-weighted returns"
        wt = market_cap.shift(1) if lag_weights else market_cap
        return _compute_weighted_quantile_df(labels, return_adj, wt, reindex=False)
    else:
        return _compute_quantile_df(labels, return_adj, reindex=False)
