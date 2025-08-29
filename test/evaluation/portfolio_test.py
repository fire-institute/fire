from firefin.evaluation.academia.portfolio_sort import PortfolioSort
from firefin.data.fake import gen_df



factor1 = gen_df(10, 100, index="day", mock="rand")
factor2 = gen_df(10, 100, index="day", mock="rand")

# 1, 2, 3, 4, 5 periods
forward_returns = { i: gen_df(10, 100, index="day", mock="return") for i in range(1, 6) }
market_cap = gen_df(10, 100, index="day", mock="volume")

def test_single_sort():
    # test single sort
    single_sort_r = PortfolioSort.single_sort(factor1, forward_returns, market_cap, quantiles=5)
    statistical_r = PortfolioSort.get_statistics(single_sort_r, quantiles=5)

def test_dual_sort():
    # test dual sort
    dual_sort_r = PortfolioSort.double_sort(factor1, factor2, forward_returns, market_cap, quantiles=(3,5))
    # statistical_r_dual = PortfolioSort.get_statistics(dual_sort_r, quantiles=5)
    print(dual_sort_r)

test_dual_sort()