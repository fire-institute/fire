from firefin.evaluation.academia.MSR_Test import MSRTest
from firefin.data.fake import gen_df

factor1 = gen_df(253, 100, index="day", mock="rand")
factor2 = gen_df(253, 100, index="day", mock="rand")

def test_MSR_Test():
    result = MSRTest.run_msr_comparison(factor1, factor2, regularize_covariance=True)
    print("Model A MSR:", result["msr_a"])
    print("Model B MSR:", result["msr_b"])
    print("Z-statistic:", result["test_stat"])
    print("P-value:", result["p_value"])

test_MSR_Test()