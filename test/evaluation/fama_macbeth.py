from firefin.evaluation.academia.fama_macbeth import FamaMacBeth
from firefin.data.fake import gen_df
from firefin.data import fetch_data

data = fetch_data(['open','close','volume','return_adj'])

def test_fama_macbeth_regression():
    r = FamaMacBeth.run_regression(data['close'], data['return_adj'], window=252, verbose=10, n_jobs=24)
    print(r)
    stats = FamaMacBeth.test_statistics(r)
    print(stats)

test_fama_macbeth_regression()