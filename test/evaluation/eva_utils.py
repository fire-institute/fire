from firefin.data.fake import gen_df
from firefin.evaluation.eva_utils import factor_to_quantile_dependent_double_sort

def test_factor_to_quantile_dependent_double_sort():
    factor1 = gen_df(10, 100, index="day", mock="rand")
    factor2 = gen_df(10, 100, index="day", mock="rand")

    double_sort = factor_to_quantile_dependent_double_sort(factor1, factor2, quantiles=(3, 5))
    print(double_sort.head())


test_factor_to_quantile_dependent_double_sort()