from firefin.data import fetch_data
from firefin.evaluation.eva_utils import compute_forward_returns
from firefin.compute.window import ts_corr
from firefin.evaluation.industry import Evaluator
from firefin.common.config import logger
import pandas as pd
import numpy as np
# fetch data
data = fetch_data( ['open','close','volume','return_adj'])

# compute pv correlation
def pv_corr(close, volume):
    return ts_corr(close, volume, 5, method="pearson")

# compute factor
factor = pv_corr(data["close"], data["volume"])

# compute forward returns
fr = compute_forward_returns(data["open"].shift(-1), [1, 5, 10])

# compute academia evaluation
from firefin.evaluation.academia.AcaEvaluatorModel import AcaEvaluatorModel

mng = AcaEvaluatorModel(factor=factor, forward_returns=fr, return_adj=data["return_adj"], n_jobs=24, verbose=10)

#test run single sort
def test_run_single_sort():
    results = {}
    results['single_sort_res'], results['single_sort_stat'] = mng.run_single_sort(
        quantiles=5,
        value_weighted=False,
        return_stats=True
    )
    print(results['single_sort_res'])
    print(results['single_sort_stat'])
test_run_single_sort()
logger.info("Test Single Sort Completed")

#test run fama macbeth
def test_run_fama_macbeth():
    results = {}
    results['fama_macbeth_res'], results['fama_macbeth_stat'] = mng.run_fama_macbeth(
        window=252, return_stats=True)
    print(results['fama_macbeth_res'])
    print(results['fama_macbeth_stat'])
test_run_fama_macbeth()
logger.info("Test Fama-MacBeth Regression Completed")

#test run ic
def test_run_ic():
    results = {}
    results['information_coefficient'] = mng.run_ic(method="pearson")
    print(results['information_coefficient'])
test_run_ic()
logger.info("Test IC Completed")

#test run regression
def test_run_regression():
    results = {}
    results['regression'] = mng.run_regression(rolling=False, fit_intercept=True)
    print(results['regression'])
test_run_regression()
logger.info("Test Static Regression Completed")

#test run anomaly test
def test_run_anomaly_test():
    results = {}
    results['single_sort_res'], results['single_sort_stat'] = mng.run_single_sort(
        quantiles=5,
        value_weighted=False,
        return_stats=True
    )
    for k, v in results['single_sort_res'].items():
        results['anomaly_stat'] = {k:mng.run_anomaly_test(portfolio_returns= pd.DataFrame(v.iloc[:,-1]), return_stats= True)}
    print(results['anomaly_stat'])
test_run_anomaly_test()
logger.info("Test Anomaly Test Completed")

#test run all
def test_run_all():
    results = mng.run_all()
    print(results)
test_run_all()
logger.info("Test run all Completed")


