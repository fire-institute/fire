# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt


from firefin.data import fetch_data
from firefin.evaluation.eva_utils import compute_forward_returns
from firefin.compute.window import ts_corr
from firefin.evaluation.industry import Evaluator
# fetch data
data = fetch_data(['open','close','volume','return_adj'])

# compute pv correlation
def pv_corr(close, volume):
    return ts_corr(close, volume, 5, method="pearson")

# compute factor
factor = pv_corr(data["close"], data["volume"])

# compute forward returns
fr = compute_forward_returns(data["open"].shift(-1), [1, 5, 10])

# compute industry evaluation
mng = Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)

# compute academia evaluation
from firefin.evaluation.academia.AcaEvaluatorModel import AcaEvaluatorModel

mng = AcaEvaluatorModel(factor=factor, forward_returns=fr, return_adj=data["return_adj"], n_jobs=24, verbose=10)
mng.run_all()