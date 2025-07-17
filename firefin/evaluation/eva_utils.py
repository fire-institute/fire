# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

# TODO: Move some common algorithms to fire/core/algorithm/

import typing
from typing import List, Optional
import numpy as np
import pandas as pd
import re

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

def _format_df_cols(
        df: pd.DataFrame, 
        percent_cols: List[int] = None, 
        bracket_cols: List[int] = None
) -> pd.DataFrame:
    """
    Format specific columns in a DataFrame:
    - Columns in `percent_cols` will be formatted as percentages (e.g., '12.34%')
    - Columns in `bracket_cols` will be formatted as bracketed values (e.g., '(12.34)')
    - All other numeric columns will be formatted to 2 decimal places (e.g., '12.34')

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to format.
    percent_cols : list of int, optional
        List of column indices to format as plain percentages.
    bracket_cols : list of int, optional
        List of column indices, formatted as value with brackets.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with specified columns formatted as strings.
    """
    formatted_df = df.copy().astype(object)

    percent_cols = percent_cols or []
    bracket_cols = bracket_cols or []
    all_formatted = set(percent_cols + bracket_cols)

    for col_idx in percent_cols:
        formatted_df.iloc[:, col_idx] = formatted_df.iloc[:, col_idx].map(lambda x: f"{x:.2%}")

    for col_idx in bracket_cols:
        formatted_df.iloc[:, col_idx] = formatted_df.iloc[:, col_idx].map(lambda x: f"({x:.2})")

    # Format remaining columns to 2 decimal places
    for col_idx in range(formatted_df.shape[1]):
        if col_idx not in all_formatted:
            formatted_df.iloc[:, col_idx] = formatted_df.iloc[:, col_idx].map(lambda x: f"{x:.2f}")

    return formatted_df

def _interleave_dfs(
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        correspondence: List[List[int]],
        interleave_rows: Optional[List[int]] = None
) -> pd.DataFrame:
    '''
    Interleave two DataFrames row-wise based on column correspondence.

    This function aligns columns of `df2` to `df1` using the provided correspondence,
    then interleaves rows from both DataFrames, inserting `None` for missing indices in `df2`.

    Parameters
    ----------
    df1: pd.Dataframe
        The primary DataFrame whose structure determines the output format. 
        Its index and columns are used as the template for alignment.

    df2: pd.Dataframe
        The secondary DataFrame whose columns will be aligned to `df1`.

    correspondence: List[List[int]]
        A list of column index pairs `[i, j]`, where:
        - `i` is the column index in `df1` (and output DataFrame).
        - `j` is the column index in `df2` to map to `df1`'s column `i`.
        Example: `[[0, 1], [2, 0]]` maps `df2[:, 1]` to `df1[:, 0]`, and `df2[:, 0]` to `df1[:, 2]`.
    
    interleave_rows : Optional[List[int]], default=None
        If provided, specifies which row indices (by position) to interleave from `df1`.
        Other rows from `df1` will be appended after interleaving.
 
    Returns
    -------
    pd.DataFrame
        Interleaved DataFrame: df1 row, then df2 row (with NaNs except at mapped positions), repeated.
    '''
    if interleave_rows is None:
        interleave_rows = list(range(df1.shape[0]))
    interleave_index = df1.index[interleave_rows]

    # Create aligned version of df2
    df2_aligned = pd.DataFrame(
        index = interleave_index, 
        columns = df1.columns, 
        dtype = object
    )
    for i, j in correspondence:
        df2_aligned.iloc[:, i] = df2.iloc[:, j]

    # Add index column for LaTeX use
    df1_full = df1.copy()
    df1_full.insert(
        loc = 0, 
        column = 'Index', 
        value = df1.index
    )
    df2_full = df2_aligned.copy()
    df2_full.insert(
        loc = 0, 
        column = 'Index', 
        value = [None] * len(interleave_rows)
    )

    # Interleave selected rows
    rows = []
    for (_, row1), (_, row2) in zip(df1_full.iloc[interleave_rows].iterrows(), df2_full.iterrows()):
        rows.extend([row1, row2])
    interleaved = pd.DataFrame(rows, columns = df1_full.columns)

    # Append non-interleaved rows if any
    non_interleave_df = df1_full.drop(interleave_index)
    if not non_interleave_df.empty:
        interleaved = pd.concat([interleaved, non_interleave_df], axis = 0)

    return interleaved

# TODO: find a more proper name standing for its function
def single_sort_table1_latex(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
) -> str:
    '''
    Generate a LaTeX-formatted table from two related DataFrames (df1 and df2), 
    interleaving selected rows with standard errors and returning the final LaTeX string.
    Simply print the return value to get reproducible LaTeX code.

    Parameters
    ----------
    df1 : pd.DataFrame
        A DataFrame containing the main results for each portfolio. 
        The index should represent portfolio names.
        Columns (in order) must be:
            0: Monthly Excess Return
            1: Standard Deviation
            2: Alpha (CAPM)
            3: VWRF (CAPM)
            4: Adj R-squared (CAPM)
            5: Alpha (4-Factor Model)
            6: RMRF (4-Factor Model)
            7: SMB (4-Factor Model)
            8: HML (4-Factor Model)
            9: PR1YR (4-Factor Model)
            10: Adj R-squared (4-Factor Model)

    df2 : pd.DataFrame
        A DataFrame containing standard errors corresponding to selected columns from df1.
        The index must exactly match df1.
        The columns (in order) are:
            0: Std of Alpha (CAPM)
            1: Std of VWRF (CAPM)
            2: Std of Alpha (4-Factor Model)
            3: Std of RMRF (4-Factor Model)
            4: Std of SMB (4-Factor Model)
            5: Std of HML (4-Factor Model)
            6: Std of PR1YR (4-Factor Model)

    Returns
    -------
    str
        Full LaTeX codes as a single string
    '''
    # index = ['1A', '1B', '1C', '1 (high)'] + list(range(2, 10)) + \
    # ['10 (low)', '10A', '10B', '10C', '1-10 spread', '1A-1C spread', '9-10 spread']
    percent_cols = [[0, 1, 2, 5], []]
    bracket_cols = [[], list(range(df2.shape[1]))]
    mean_std_pairs = [[2, 0], [3, 1], [5, 2], [6, 3], [7, 4], [8, 5], [9, 6]]

    formatted_df1 = _format_df_cols(
        df1, 
        percent_cols[0], 
        bracket_cols[0]
    )
    formatted_df2 = _format_df_cols(
        df2, 
        percent_cols[1], 
        bracket_cols[1]
    )

    df3 = _interleave_dfs(
        formatted_df1, 
        formatted_df2, 
        correspondence = mean_std_pairs
    )
    
    df3_latex_code = df3.to_latex(header = False, index = False, escape = True).replace('NaN', '')
    # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
    df_latex_code = [
        line for line in df3_latex_code.splitlines()
        if not any(i in line for i in ['\\begin{tabular}', 'toprule'])
    ]

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{*{12}c}',
        r'\toprule',
        r' & & & \multicolumn{3}{c}{\multirow{2}{*}{CAPM}} & \multicolumn{6}{c}{\multirow{2}{*}{4-Factor Model}} \\',
        r' & Monthly & & \multicolumn{3}{c}{\hrulefill} & \multicolumn{6}{c}{\hrulefill} \\',
        r' & Excess & Std & & & Adj & & & & & & Adj \\ ',
        r'Portfolio & Return & Dev & Alpha & VWRF & R-sq & Alpha & RMRF & SMB & HML & PR1YR & R-sq \\',
    ] + df_latex_code + [
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: find a more proper name standing for its function
def single_sort_table2_latex(
        df: pd.DataFrame, 
) -> str:
    '''
    Generate a LaTeX-formatted table from a single DataFrame.
    Simply print the return value to get reproducible LaTeX code.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing results for each portfolio.
        The index should represent portfolio names.
        Columns (in order) must be:
            0: Excess Return
            1: Standard Deviation
            2: Alpha (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            3: Alpha-t (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            4: RMRF (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            5: SMB (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            6: HML (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            7: PR1YR (4-Factor Model Ordinary Least Squares (OLS) Estimates)
            8: Expense Ratio
            9: Turnover (Mturn)
            10: Roundtrip Transaction Costs
            11: Adjusted Alpha

    Returns
    -------
    str
        Full LaTeX codes as a single string
    '''
    # index = ['1 (high)'] + list(range(2, 10)) + ['10 (low)', '1-10 spread', '9-10 spread']

    percent_cols = [0, 1, 2, 10, 11]
    bracket_cols = [3]

    formatted_df = _format_df_cols(df, percent_cols, bracket_cols).to_latex(header = False, escape = True)

    # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
    df_latex_code = [
        line for line in formatted_df.splitlines()
        if not any(i in line for i in ['\\begin{tabular}', 'toprule'])
    ]

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{*{13}c}',
        r'\toprule',
        r' & & & \multicolumn{6}{c}{\multirow{2}{*}{4-Factor Model Ordinary Least Squares (OLS) Estimates}} & & & Roundtrip \\',
        r' & Excess & Standard & \multicolumn{6}{c}{\hrulefill} & Exp & Turn & Transaction & Adjusted \\',
        r'Portfolio & Return & Deviation & Alpha & Alpha-t & RMRF & SMB & HML & PR1YR & Ration & (Mturn) & Costs & Alpha \\',
    ] + df_latex_code + [
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: find a more proper name standing for its function
def single_sort_table3_latex(
        df1: pd.DataFrame,
        df2: pd.DataFrame
) -> str:
    '''
    Generate a LaTeX-formatted table from two related DataFrames (df1 and df2), 
    interleaving selected rows with standard errors and returning the final LaTeX string.
    Simply print the return value to get reproducible LaTeX code.

    Parameters
    ----------
        df1 : pd.DataFrame
        A DataFrame containing the main results for each portfolio. 
        The index should represent portfolio names.
        Columns (in order) must be:
            0: P1 (low beta)
            1: P2
            2: P3
            3: P4
            4: P5
            5: P6
            6: P7
            7: P8
            8: P9
            9: P10 (high beta)
            10: BAB

    df2 : pd.DataFrame
        A DataFrame containing standard errors corresponding to selected columns from df1.
        The index must exactly match df1.
        The columns (in order) are:
            0: Std of P1 (low beta)
            1: Std of P2
            2: Std of P3
            3: Std of P4
            4: Std of P5
            5: Std of P6
            6: Std of P7
            7: Std of P8
            8: Std of P9
            9: Std of P10 (high beta)
            10: Std of BAB

    Returns
    -------
    str
        Full LaTeX codes as a single string
    '''
    # index = ['Excess return', 'CAPM alpha', 'Three-factor alpha', 'Four-factor alpha', 'Five-factor alpha', 'Beta (ex ante)', 
    #      'Beta (realized)', 'Volatility', 'Sharpe ratio']

    percent_cols = [[], []]
    bracket_cols = [[], list(range(df2.shape[1]))]
    mean_std_pairs = [[i, i] for i in range(df2.shape[1])]

    formatted_df1 = _format_df_cols(df1, percent_cols[0], bracket_cols[0])
    formatted_df2 = _format_df_cols(df2, percent_cols[1], bracket_cols[1])
    df3 = _interleave_dfs(
        formatted_df1, 
        formatted_df2, 
        correspondence = mean_std_pairs,
        interleave_rows = list(range(df2.shape[0]))
    )

    df3_latex_code = df3.to_latex(header = False, index = False, escape = True).replace('NaN', '')
    # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
    df_latex_code = [
        line for line in df3_latex_code.splitlines()
        if not any(i in line for i in ['\\begin{tabular}', 'toprule'])
    ]

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{*{12}c}',
        r'\toprule',
        r'Portfolio & P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9 & P10 & BAB \\',
        r'& (low beta) & & & & & & & & & (high beta) & \\',
    ] + df_latex_code + [
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: Complete the comments about df and return
# TODO: find a more proper name standing for its function
def fama_macbeth_latex(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df3: pd.DataFrame,
        df4: pd.DataFrame
) -> str:  
    '''
    Generate a LaTeX-formatted side-by-side table for Fama-MacBeth regression results.

    This function formats and interleaves two pairs of DataFrames, then generates LaTeX 
    code to display them in two vertically stacked tables arranged side by side.

    Parameters
    ----------
    df1 : pd.DataFrame
        The index should represent variable (e.g., 'CGO', 'FROXY', 'PROXY x CGO').
        Columns (in order) must be:
            0: Variable (1)
            1: Variable (2)
            2: Variable (3)
            3: Variable (4)

    df2 : pd.DataFrame
        The index must exactly match df1.
        The columns (in order) are:
            0: Variable (1)
            1: Variable (2)
            2: Variable (3)
            3: Variable (4)

    df3 : pd.DataFrame
        The index should represent variable (e.g., 'CGO', 'FROXY', 'PROXY x CGO').
        Columns (in order) must be:
            0: Variable (1)
            1: Variable (2)
            2: Variable (3)
            3: Variable (4)
        
    df4 : pd.DataFrame
        The index must exactly match df3.
        The columns (in order) are:
            0: Variable (1)
            1: Variable (2)
            2: Variable (3)
            3: Variable (4)

    Returns
    -------
    str
        A string of LaTeX code representing two side-by-side tables
    '''
    # index = ['CGO', 'FROXY', 'PROXY x CGO', '\\multirow{2}{*}{\\shortstack[l]{PROXY x \\\\ MOM(-12, -1)}}', 
    #          'MOM(-1, 0)', 'MOM(-12, -1)', 'TURNOVER']
    percent_cols = [[], []]
    bracket_cols = [[], list(range(df2.shape[1]))]
    mean_std_pairs = [[i, i] for i in range(df2.shape[1])]

    latex_list = []
    df_dict = {1: df1, 2: df2, 3: df3, 4: df4}
    for i in [1, 3]:
        formatted1 = _format_df_cols(df_dict[i], percent_cols[0], bracket_cols[0])
        formatted2 = _format_df_cols(df_dict[i + 1], percent_cols[1], bracket_cols[1])
        df_latex_code = _interleave_dfs(
            formatted1, 
            formatted2, 
            correspondence = mean_std_pairs
        ).to_latex(header = False, index = False)
        df_latex_code = re.sub('\(?nan\)?', '', df_latex_code, flags = re.IGNORECASE)
        # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
        df_latex_code = [
            line for line in df_latex_code.splitlines()
            if not any(i in line for i in ['\\begin{tabular}', 'toprule'])
        ]
        latex_list.append(df_latex_code)

    latex_code = [
        r'\begin{table}[ht]', 
        r'\centering',
        r'\begin{minipage}{0.48\textwidth}',
        r'\centering',
        r'\begin{tabular}{l*{4}c}',
        r'\toprule',
        r'\toprule',
        r'Variable & (1) & (2) & (3) & (4) \\',
    ] + latex_list[0] + [
        r'\end{minipage}', 
        r'\hfill',
        r'\begin{minipage}{0.48\textwidth}',
        r'\centering',
        r'\begin{tabular}{l*{4}c}',
        r'\toprule',
        r'\toprule',
    ] + latex_list[1] + [
        r'\end{minipage}',
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: Complete the comments about df and return
# TODO: find a more proper name standing for its function
def regression_latex(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df3: pd.DataFrame,
        df4: pd.DataFrame
) -> str:
    '''
    Generate a LaTeX-formatted table of interleaved regression results.

    This function takes two pairs of regression outputs, interleaves them 
    row-wise, and formats them into a LaTeX tabular structure with predefined panel headings.

    Parameters
    ----------
    df1 : pd.DataFrame
        Columns (in order) must be:
            0 and 1: Global equities
            1 and 2: F1 10Y global
            3 and 4: F1 10Y-2Y global
            5 and 6: US Treasuries
            7 and 8: Commodities

    df2 : pd.DataFrame
        The index must exactly match df1.
        The columns (in order) are:
            0 and 1: Global equities
            1 and 2: F1 10Y global
            3 and 4: F1 10Y-2Y global
            5 and 6: US Treasuries
            7 and 8: Commodities

    df3 : pd.DataFrame
        Columns (in order) must be:
            0 and 1: Currencies
            1 and 2: Credits
            3 and 4: Call options
            5 and 6: Put options
            7 and 8: GCF
        
    df4 : pd.DataFrame
        The index must exactly match df1.
        The columns (in order) are:
            0 and 1: Currencies
            1 and 2: Credits
            3 and 4: Call options
            5 and 6: Put options
            7 and 8: GCF

    Returns
    -------
    str
        A string of LaTeX code representing a formatted table
    '''
    # index = ['$\\alpha$', 'Passive long', 'Value', 'Momentum', 'TSMOM', '$R^2$', 'IR']
    percent_cols = [[], []]
    bracket_cols = [[], list(range(df2.shape[1]))]
    mean_std_pairs = [[i, i] for i in range(df2.shape[1])]

    latex_list = []
    df_dict = {1: df1, 2: df2, 3: df3, 4: df4}
    for i in [1, 3]:
        formatted1 = _format_df_cols(df_dict[i], percent_cols[0], bracket_cols[0])
        formatted2 = _format_df_cols(df_dict[i + 1], percent_cols[1], bracket_cols[1]) 
        df_latex_code = _interleave_dfs(
            formatted1, 
            formatted2, 
            correspondence = mean_std_pairs, 
            interleave_rows = list(range(df2.shape[0]))
        ).to_latex(header = False, index = False)
        df_latex_code = re.sub('\(?nan\)?', '', df_latex_code, flags = re.IGNORECASE)
        # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
        df_latex_code = [
            line for line in df_latex_code.splitlines()
            if not any(i in line for i in ['\\begin{tabular}', 'toprule', '\\end{tabular}'])
        ]
        latex_list.append(df_latex_code)

    latex_code = [
        r'\begin{table}[ht]', 
        r'\centering',
        r'\begin{tabular}{*{11}c}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Global equities} & \multicolumn{2}{c}{F1 10Y global} &' + \
            r'\multicolumn{2}{c}{F1 10Y-2Y global} & \multicolumn{2}{c}{US Treasuries} &' + \
            r'\multicolumn{2}{c}{Commodities} \\'
    ] + latex_list[0] + [
        r'& \multicolumn{2}{c}{Currencies} & \multicolumn{2}{c}{Credits} &' + \
            r'\multicolumn{2}{c}{Call options} & \multicolumn{2}{c}{Put options} &' + \
            r'\multicolumn{2}{c}{GCF} \\'
    ] + latex_list[1] + [
        r'\end{tabular}', 
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: Complete the comments about df and return
# TODO: find a more proper name standing for its function
def else1_latex(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    '''
    Generate LaTeX code for a two-panel table.

    This function takes two DataFrames: one for baseline models and one for FS-FMB procedure.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame representing Panel A: baseline model.
        The index should represent serial number.
        Columns (in order) must be:
            0: # standing for serial number
            1: Model
            2: Adj.R-squared 
            3: alpha
            4: t-stat(alpha)

    df2 : pd.DataFrame
        DataFrame representing Panel B: FS-FMB procedure.
        The index should represent step number.
        Columns (in order) must be:
            0: Step
            1: h_j
            2: Adj.R-squared 
            3: alpha
            4: t-stat(alpha)

    Returns
    -------
    str
        A LaTeX string for rendering a two-panel regression summary table.
    '''
    # Model = ['CAPM', 'FF3', 'FF5', 'FF5M']
    # h_j = ['SMB2', 'SMB2*Mom', 'Mom2*RMW', 'Mkt-RF2','Mkt-RF2*RMW', 'Mkt-Rf*SMB', 'HML2*Mkt-RF']
    latex_list = []
    df_dict = {1: df1, 2: df2}
    for i in [1, 2]:
        df_latex_code = df_dict[i].to_latex(escape = True, header = False, float_format = '%.3f')
        # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
        df_latex_code = [
            line for line in df_latex_code.splitlines()
            if not any(i in line for i in ['\\begin{tabular}', 'toprule', '\\end{tabular}'])
        ]
        latex_list.append(df_latex_code)

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{clccc}',
        r'\toprule',
        r'\multicolumn{5}{c}{Panel A: Baseline Models} \\',
        r'\midrule',
        r'\# & Model & Adj.R-squared & $\alpha$ & t-stat($\alpha$) \\'
    ] + latex_list[0] + [
        r'\multicolumn{5}{c}{Panel B: FS-FMB procedure} \\',
        r'\midrule',
        r'Step & $h_j$ & Adj.R-squared & $\alpha$ & t-stat($\alpha$) \\',
    ] + latex_list[1] + [
        r'\end{tabular}', 
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code 

# TODO: Complete the comments about df and return
# TODO: find a more proper name standing for its function
def else2_latex(df: pd.DataFrame) -> str:
    '''
    Generate LaTeX code for a regression R-squared comparison table.

    Parameters
    ----------
    df : pd.DataFrame
        The index should represent various R-square.
        Columns (in order) must be:
            0: CAPM
            1: FF3
            2: FF5
            3: FF5M
            4: Higher-Order

    Returns
    -------
    str
        A LaTeX-formatted string representing a two-row table for model R² comparison.
    '''
    # ['$R_{train}^2$', '$R_{oos}^2$']
    df_latex_code = df.to_latex(header = False, float_format = '%.3f')
    # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
    df_latex_code = [
        line for line in df_latex_code.splitlines()
        if not any(i in line for i in ['\\begin{tabular}', 'toprule'])
    ]

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{*{6}c}',
        r'\toprule',
        r'& (1) & (2) & (3) & (4) & (5) \\',
        r'& CAPM & FF3 & FF5 & FF5M & Higher-Order\\'
    ] + df_latex_code + [
        r'\end{table}'
    ]
    latex_code = '\n'.join(latex_code)

    return latex_code

# TODO: Complete the comments about df and return
# TODO: find a more proper name standing for its function
def else3_latex(
        df: pd.DataFrame, 
        val1: float, 
        val2: float
    ) -> str:
    '''
    Generate LaTeX code for a table showing factors.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with index representing factor names.
        The index should represent factor.
        Columns (in order) must be:
            0: Frac Sig 5% (1)
            1: Frac Sig 5% (2)

    val1 : float

    val2 : float

    Returns
    -------
    str
        A LaTeX-formatted table string.
    '''
    # index = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'SMB2', 'SMB2*Mom', 
    #          'Mom2*RMW', 'Mkt-RF2', 'Mkt-RF2*RMW', 'Mkt-RF*SMB', 'HML*2Mkt-RF']
    df_latex_code = df.to_latex(header = False, float_format = '%.3f')
    df_latex_code = re.sub('\(?nan\)?', '', df_latex_code, flags = re.IGNORECASE)
    # ['\\begin{tabular}', 'toprule'] have been typeset so deleted
    df_latex_code = [
        line for line in df_latex_code.splitlines()
        if not any(i in line for i in ['\\begin{tabular}', 'toprule', '\\end{tabular}'])
    ]

    latex_code = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'& (1) & (2) \\',
        r'\midrule',
        r'Factor & Frac Sig 5\% & Frac Sig 5\% \\'
    ] + df_latex_code + [
        f'\\# zoo factors & {val1} & {val2} \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    latex_code = '\n'.join(latex_code)

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

def _compute_quantile_df(
        qt: pd.DataFrame, 
        fr: pd.DataFrame, 
        reindex = True, 
        quantiles: int = 5
) -> pd.DataFrame:
    '''
    Compute equal-weighted average forward returns for each quantile group.

    Assumes that `qt` (quantile assignments) and `fr` (forward returns) are aligned
    by index and columns — i.e., same dates (index) and same stocks (columns).

    Parameters
    ----------
    qt : pd.DataFrame
        Quantile assignment for each asset at each time.
        Index: time, Columns: stock code, Values: quantile group (int from 1 to `quantiles`)
    
    fr : pd.DataFrame
        Forward returns for each asset at each time.
        Index: time, Columns: stock code, Values: future return

    reindex : bool, default True
        Whether to ensure the result has columns 1 to `quantiles` (even if some are missing at certain times)

    quantiles : int, default 5
        Number of quantile groups

    Returns
    -------
    pd.DataFrame
        A time-series DataFrame of average returns for each quantile group.
        Index: time, Columns: quantile group (1 ~ `quantiles`)
    '''
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row) in zip(fr.iterrows(), qt.iterrows()):
        result[dt] = fr_row.groupby(qt_row).mean()
    result = pd.DataFrame(result).T
    if reindex:
        return result.reindex(columns = np.arange(1, quantiles + 1), copy = False)
    return result

def _compute_weighted_quantile_df(
        qt: pd.DataFrame, 
        fr: pd.DataFrame, 
        wt: pd.DataFrame, 
        reindex = True, 
        quantiles: int = 5
) -> pd.DataFrame:
    '''
    Compute value-weighted average forward returns for each quantile group.

    This is the weighted version of `_compute_quantile_df`, where the group-wise 
    mean is computed using market capitalization.

    Assumes that `qt`, `fr`, and `wt` are aligned by index and columns:
    i.e., same dates (index) and same stocks (columns).

    Parameters
    ----------
    qt : pd.DataFrame
        Quantile assignment for each asset at each time.
        Index: time, Columns: stock code, Values: quantile group (int from 1 to `quantiles`)

    fr : pd.DataFrame
        Forward returns for each asset at each time.
        Index: time, Columns: stock code, Values: forward return

    wt : pd.DataFrame
        Value weights for each asset at each time (e.g., market capitalization).
        Index: time, Columns: stock code, Values: weight

    reindex : bool, default True
        Whether to ensure the result has columns 1 to `quantiles` 
        (even if some quantile groups are missing at some timestamps)

    quantiles : int, default 5
        Number of quantile groups

    Returns
    -------
    pd.DataFrame
        A time-series DataFrame of value-weighted average returns for each quantile group.
        Index: time, Columns: quantile group (1 ~ `quantiles`)
    '''
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
