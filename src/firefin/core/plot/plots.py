# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec
from numba import njit
from scipy import stats
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from ...core.eva_utils import PeriodType, IC, QuantileReturns

__all__ = [
    "plt_ic",
    "plt_cumulative_returns",
    "plt_quantile_cumulative_returns",
    "plt_quantile_cumulated_end_returns",
]


sns.set_style("whitegrid")


def _plt_cumsum_ic(summarized_data, ax, factor_name, data_name):
    y_mean = summarized_data.resample("YE", label="left").mean()
    for c, sr in y_mean.T.iterrows():
        ax.scatter(sr.index, sr, marker="x")
    ax.set_title(f"{factor_name} Cumulative {data_name} & Yearly Mean")
    ax.axhline(0, linestyle="-", color="black", lw=1)

    axr = ax.twinx()
    axr.plot(summarized_data.cumsum())
    axr.legend(summarized_data.columns, loc=2)
    axr.grid(linestyle=":")

    # make sure the left axis has visible 0-line
    b, t = ax.get_ylim()
    if b > 0:
        ax.set_ylim(0, t)
    elif t < 0:
        ax.set_ylim(b, 0)


def _summarize_ic_data(data):
    origin_columns = data.columns

    _mean = data.mean()
    _std = data.std()
    _ir = data.mean() / data.std()

    summary_columns = [f"{c}, AVG={_mean[c]:.2%}, STD={_std[c]:.2%}, IR={_ir[c]:.2f}" for c in origin_columns]

    summarized_data = data.rename(columns=dict(zip(origin_columns, summary_columns)))

    return summarized_data


def _plt_monthly_and_20ma_ic(data, axs, data_name, color_bounds):
    origin_columns = data.columns
    markersize = 5

    data_month = data.resample("ME").mean()
    summarized_data = _summarize_ic_data(data)

    for i_p, col in enumerate(origin_columns):
        ax = axs[i_p]

        data_col = summarized_data.iloc[:, i_p]
        ax.plot(data_col.rolling(20, min_periods=1).mean())

        month_p_data = data_month.iloc[:, i_p]

        for color, bounds in color_bounds.items():
            data_sel = month_p_data[(month_p_data >= bounds[0]) & (month_p_data <= bounds[1])]
            data_sel.plot(color=color, linestyle="", marker="D", markersize=markersize, ax=ax)

        ax.set(
            xlabel="",
            title=f"{col} {data_name}, Monthly Average and 20-day MA",
        )

    return summarized_data


def plt_ic(ic_data: IC, factor_name="Factor", dist=True, plot_dir=None, show=True):
    """
    Plot an IC plot with Monthly IC, Cumulative IC and IC distribution.

    Parameters
    ----------
    ic_data: IC
    factor_name: str
    plot_dir: None or Path
    show: bool
    dist: bool
        if True, show distribution of IC and its QQ-plot.

    """

    ic_data = ic_data.dropna(how="all")
    if not isinstance(ic_data.index, pd.DatetimeIndex):
        ic_data.index = pd.DatetimeIndex(ic_data.index)
    columns = ic_data.columns
    n_cols = len(columns)

    if dist:
        fig_width = 20
        grid_width = 4
    else:
        fig_width = 10
        grid_width = 2

    fig = plt.figure(figsize=(fig_width, 3.5 * n_cols))
    grid = GridSpec(n_cols * 4, grid_width, figure=fig)

    # 折线图，每个period一张，高度为3
    ax0 = fig.add_subplot(grid[:3, :2])
    axs = [ax0]
    for i_p in range(1, n_cols):
        axs.append(fig.add_subplot(grid[i_p * 3 : (i_p + 1) * 3, :2], sharex=ax0, sharey=ax0))
    # 累计图，一张，高度为n_periods
    axs.append(fig.add_subplot(grid[n_cols * 3 :, :2]))

    if dist:
        # 分布图，每个period一张，高度为4
        for i_p in range(n_cols):
            for i in range(2):
                axs.append(fig.add_subplot(grid[i_p * 4 : (i_p + 1) * 4, 2 + i]))

    # matplotlib/_color_data.py
    # https://drafts.csswg.org/css-color-4/#named-colors
    # 都是闭区间，先画暗色，再画两端的亮色以及0的灰色
    color_bounds = {
        "grey": [-0.02, 0.02],
        "darkblue": [-0.05, -0.02],
        "darkred": [0.02, 0.05],
        "blue": [-np.inf, -0.05],
        "red": [0.05, np.inf],
    }

    summarized_data = _plt_monthly_and_20ma_ic(ic_data, axs, "IC", color_bounds=color_bounds)
    _plt_cumsum_ic(summarized_data, axs[n_cols], factor_name, "IC")

    if dist:
        for i_p, p in enumerate(columns):
            ic_data_p = ic_data.iloc[:, i_p].dropna()

            ax1, ax2 = axs[n_cols + 1 + i_p * 2], axs[n_cols + 2 + i_p * 2]

            sns.histplot(ic_data_p, kde=True, bins=int(np.ceil(np.log(ic_data_p.size) * 10)), stat="density", ax=ax1)
            ax1.set(
                xlabel=f"{p}, Mean {ic_data_p.mean():.2f}, Skew {ic_data_p.skew():.2f}, Kurt {ic_data_p.kurt():.2f}"
            )
            sm.qqplot(ic_data_p, stats.norm, fit=True, line="45", ax=ax2)
            ax2.set(ylabel="Observed Quantile", xlabel="Norm Distribution Quantile")

    if plot_dir:
        plt.savefig(Path(plot_dir) / f"{factor_name} IC plot.png", bbox_inches="tight")
    if show:
        plt.show()

    summary_table = pd.DataFrame(
        np.nan,
        index=["mean", "std", "ir", "> 0", "< 0", "> 3%", "< -3%", "> 5%", "< -5%"],
        columns=columns,
    )

    ic_mean = ic_data.mean()
    ic_std = ic_data.std()
    ir = ic_mean / ic_std

    summary_table.loc["mean"] = ic_mean.values
    summary_table.loc["std"] = ic_std.values
    summary_table.loc["ir"] = ir.values
    summary_table.loc["> 0"] = ((ic_data > 0).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< 0"] = ((ic_data < 0).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["> 3%"] = ((ic_data > 0.03).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< -3%"] = ((ic_data < -0.03).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["> 5%"] = ((ic_data > 0.05).sum() / np.isfinite(ic_data).sum()).values
    summary_table.loc["< -5%"] = ((ic_data < -0.05).sum() / np.isfinite(ic_data).sum()).values
    print(summary_table)


def _get_annual_and_end_returns(daily_cum_returns):
    daily_cum_returns = np.asarray(daily_cum_returns)
    n, m = daily_cum_returns.shape
    na_mask = np.isfinite(daily_cum_returns)
    end_returns = []

    for j in range(m):
        for i in range(n):
            if na_mask[n - 1 - i, j]:
                end_returns.append(daily_cum_returns[n - 1 - i, j])
                break
    end_returns = np.asarray(end_returns)
    annual_returns = np.float_power(np.add(end_returns, 1), 244 / na_mask.sum(axis=0)) - 1
    return annual_returns, end_returns


def plt_cumulative_returns(
    *,
    daily_returns=None,
    daily_cum_returns=None,
    show_min_max=True,
    title="Cumulative Returns",
    ax=None,
    show=False,
    plot_dir=None,
):
    """

    Parameters
    ----------
    daily_returns: pd.DataFrame
    daily_cum_returns: pd.DataFrame
    show_min_max: bool
    title: str
    ax: matplotlib axis
    show: bool
    plot_dir: Path, default=None

    """

    if daily_returns is None:
        if daily_cum_returns is None:
            raise ValueError(f"one of daily_returns or daily_cum_returns must be provided")
        else:
            daily_cum_returns = daily_cum_returns.dropna(how="all")
            daily_returns = daily_cum_returns.add(1).pct_change()
            daily_returns.iloc[0] = daily_cum_returns.iloc[0]

    else:
        if daily_cum_returns is not None:
            raise ValueError(f"exactly one of daily_returns or daily_cum_returns should be provided")
        else:
            daily_returns = daily_returns.dropna(how="all")
            daily_cum_returns = daily_returns.add(1).cumprod() - 1

    # set name for columns
    annual_returns, end_returns = _get_annual_and_end_returns(daily_cum_returns)
    daily_cum_returns.columns = [
        f"{c}, ANN. {art:.2%}, TOT. {ert:.2%}"
        for c, art, ert in zip(daily_cum_returns.columns, annual_returns, end_returns)
    ]
    daily_returns.columns = daily_cum_returns.columns

    if ax is None:
        _, ax = plt.subplots()
    daily_cum_returns.plot(cmap=plt.cm.coolwarm, ax=ax)
    if show_min_max:
        max_group = daily_returns.columns[[0, -1]][np.argmax(annual_returns[[0, -1]])]
        min_group = daily_returns.columns[[0, -1]][np.argmin(annual_returns[[0, -1]])]
        min_max_diff = (daily_returns.loc[:, max_group] - daily_returns.loc[:, min_group] + 1).cumprod() - 1
        [annual_returns], [end_returns] = _get_annual_and_end_returns(min_max_diff.to_frame())
        min_max_diff.name = f"Min Max, ANN. {annual_returns:.2%}, TOT. {end_returns:.2%}"
        min_max_diff.plot(lw=2, color="black", alpha=0.8, ax=ax)

    ax.set(xlabel="", ylabel="Cumulative Returns", title=title)
    ax.legend(loc=2, ncol=int(np.ceil(len(daily_returns.columns) / 25)), fontsize=8)
    ax.axhline(0.0, linestyle="-", color="black", lw=1)

    # if logy:
    #     from matplotlib.ticker import FuncFormatter
    #
    #     log_return_locator_cls = get_log_return_locator()
    #
    #     fwd, ivt = lambda x: np.log1p(x), lambda x: np.exp(x) - 1
    #     ax.set_yscale("function", functions=(fwd, ivt))
    #     ax.set_ylim([np.exp(np.log(1 + np.nanmin(daily_cum_returns)) * 1.1) - 1, None])
    #     ax.yaxis.set_major_locator(log_return_locator_cls(base=10, linthresh=1))
    #     ax.yaxis.set_major_formatter(FuncFormatter(log_return_formater))

    if plot_dir:
        plt.savefig(plot_dir / f"{title}.png", bbox_inches="tight")
    if show:
        plt.show()


def return_to_daily(data: pd.Series | pd.DataFrame, period: PeriodType):
    """Convert period returns to daily returns."""
    if period == 1:
        return data.copy(deep=False)
    return ((data + 1) ** (1 / period)) - 1


def compute_cum_returns(daily_ret: pd.Series | pd.DataFrame):
    return (1 + daily_ret).cumprod() - 1


def _can_plot_recent(data: pd.Series | pd.DataFrame, years=3) -> tuple[bool, pd.Timestamp]:
    """check if longer than 3 years and return the -3 year loc if possible"""
    # index of this is datetime index
    index = data.index
    if (index[-1] - index[0]).days // 365 >= years:
        plot_recent = True
        loc = index[-1] - pd.Timedelta(days=365 * years)
    else:
        plot_recent = False
        loc = None
    return plot_recent, loc


def plt_quantile_cumulative_returns(quantile_returns: QuantileReturns, factor_name="Factor", plot_dir=None, show=True):
    """
    Plot the cumulative returns of each quantile.

    Parameters
    ----------
    quantile_returns: QuantileReturns
    factor_name: str
    plot_dir: Path, default None
    show: bool, default True

    """
    cum_returns = {
        period: compute_cum_returns(return_to_daily(period_returns, period))
        for period, period_returns in quantile_returns.items()
    }
    periods = sorted(cum_returns.keys())

    plot_recent, loc = _can_plot_recent(next(iter(quantile_returns.values())))

    fig, axs = plt.subplots(len(periods), 1 + plot_recent, figsize=(10 + (3 * plot_recent), 7 * len(periods)))
    for (period, period_cum_returns), ax in zip(cum_returns.items(), axs):
        if plot_recent:
            ax1, ax2 = ax
        else:
            ax1, ax2 = ax, NotImplemented

        plt_cumulative_returns(
            daily_cum_returns=period_cum_returns,
            ax=ax1,
            show_min_max=True,
            title=f"{factor_name} ({period} Fwd Period)",
            show=False,
        )
        if plot_recent:
            recent_data = period_cum_returns.loc[loc:] + 1
            recent_data = recent_data.pct_change(fill_method=None).add(1).cumprod().sub(1)
            plt_cumulative_returns(
                daily_cum_returns=recent_data,
                ax=ax2,
                show_min_max=True,
                title=f"{factor_name} (Recent) ({period} Fwd Period)",
                show=False,
            )
    if plot_dir:
        plt.savefig(
            Path(plot_dir) / f"{factor_name} Quantile Cum Returns.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()


@njit
def get_cum_end_returns(daily_rt):
    started = False
    cum_rt = 1
    total = 0
    n_cs_nans = 0
    for x in daily_rt:
        if np.isfinite(x):
            n_cs_nans = 0
            started = True
            cum_rt *= 1 + x
            total += 1
        else:
            n_cs_nans += 1
            if started:
                total += 1
    if not started:
        return np.nan
    total -= n_cs_nans
    return cum_rt ** (244 / total) - 1


def get_cumulated_end_returns(daily_ret: pd.Series | pd.DataFrame, std=False):
    """
    Get cumulated end returns of each quantile

    Parameters
    ----------
    std: bool, default False
        If True, returns the standard deviation of the cumulated end returns

    """

    returns_avg = daily_ret.apply(get_cum_end_returns, raw=True)

    if std:
        returns_std = daily_ret.std() * np.sqrt(244)
        return returns_avg, returns_std
    else:
        return returns_avg


def _get_avg_and_std(quantile_returns: QuantileReturns):
    returns_avg = {}
    returns_std = {}
    for period, period_returns in quantile_returns.items():
        returns_avg[period], returns_std[period] = get_cumulated_end_returns(period_returns, std=True)

    # quantile x periods
    return pd.DataFrame(returns_avg), pd.DataFrame(returns_std)


def plt_quantile_cumulated_end_returns(
    quantile_returns: QuantileReturns, factor_name="Factor", plot_dir=None, show=True
):
    """
    Plot the cumulated end returns of each quantile.

    Parameters
    ----------
    quantile_returns: QuantileReturns
    factor_name: str
    plot_dir: Path, default None
    show: bool, default True

    """
    returns_avg, returns_std = _get_avg_and_std(quantile_returns)

    plot_recent, loc = _can_plot_recent(next(iter(quantile_returns.values())))

    w, h = (4 * len(returns_avg) * (1 + plot_recent) + 50) / 9, 16

    def _plot(avg, std, axavg, axstd, name):
        avg.plot(kind="bar", width=0.8, ax=axavg)
        axavg.set(
            xlabel="",
            ylabel="Return Mean (Ann.)",
            title=f"{name} Return Mean By Quantile",
        )
        std.plot(kind="bar", width=0.8, ax=axstd)
        axstd.set(
            xlabel="",
            ylabel="Return Std (Ann.)",
            title=f"{name} Return Std By Quantile",
        )

    fig, axs = plt.subplots(2, 1 + plot_recent, figsize=(w, h))
    if plot_recent:
        (ax_avg1, ax_avg2), (ax_std1, ax_std2) = axs
    else:
        ax_avg1, ax_std1 = axs
        ax_avg2 = ax_std2 = None
    _plot(returns_avg, returns_std, ax_avg1, ax_std1, factor_name)

    if plot_recent:
        returns_avg_rct, returns_std_rct = _get_avg_and_std(
            QuantileReturns({k: v.loc[loc:] for k, v in quantile_returns.items()})
        )
        _plot(returns_avg_rct, returns_std_rct, ax_avg2, ax_std2, f"{factor_name} (Recent)")

    for ax in axs.flatten():
        ax.yaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))

    if plot_dir:
        plt.savefig(
            Path(plot_dir) / f"{factor_name} Quantile End Returns.png",
            bbox_inches="tight",
        )
    if show:
        plt.show()

def plot_grs_pval(series: pd.Series, threshold: float = 0.95, ax=None):
    """
    Pass in the pd.Series time series grs p value , color it according to the threshold(0,95) and draw a reference line
    params
    ----
    series : pd.Series

    threshold : float
        default is 0.95
    ax : matplotlib.axes.Axes | None

    return
    ----
    ax : matplotlib.axes.Axes
    """
    s = pd.Series(series).copy()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    s = s.sort_index().astype(float)


    mask_above = s > threshold
    s_above = s.where(mask_above)
    s_below = s.where(~mask_above)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.axhline(threshold, color="blue", linestyle="--", linewidth=1)

    ax.plot(s_below.index, s_below.values, color="black", linewidth=1)
    ax.plot(s_above.index, s_above.values, color="red", linewidth=1)

    ax.set_xlabel("time")
    ax.set_ylabel("grs p value")
    ax.set_title("grs p value")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)

    try:
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
        locator = AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    except Exception:
        pass

    return ax


def plot_cumulative_alpha(series: pd.Series, title: str = None, figsize=(10, 4), block: bool = True):
    """
    传入索引为时间的 pd.Series，画出时间-数值折线图并返回 Figure。

    params
    ----
    series : pd.Series
        time series cumulative alpha
    title : str, optional
        factor name

    return
    ----
    fig
    """
    # 复制并确保索引为 datetime
    s = pd.Series(series).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.sort_index()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, linewidth=1.2)

    # 轴与网格
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "cumulative_alpha")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)

    # 更友好的时间刻度
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    fig.tight_layout()
    plt.show(block=block)
