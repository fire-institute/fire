from unittest import TestCase
import numpy as np
import pandas as pd

from firefin.evaluation.academia.AcaEvaluatorModel import AcaEvaluatorModel

class TestAcaEvaluatorModel(TestCase):

    def test_aca_evaluator_model(self):
        factor_portfolio = [pd.Series(np.random.randn(100)), pd.Series(np.random.randn(100))]
        return_adj = pd.DataFrame(np.random.randn(100, 10))
        model = AcaEvaluatorModel(factor_portfolio, return_adj)
        model.run_time_series_regression(fit_intercept=True)
        x = model.time_series_res.beta
        print(x)