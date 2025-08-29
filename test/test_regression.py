# -*- coding: utf-8 -*-
# @Created : 2025/3/27 11:36
# @Author  : Liao Renjie
# @Email   : liao.renjie@techfin.ai
# @File    : test_regression.py
# @Software: PyCharm

from unittest import TestCase

import numpy as np
import pandas as pd
from numpy import array, nan

from firefin.core.algorithm.regression import RollingRegressor, table_regression, rolling_regression
from firefin.data.fake import gen_df


class TestRegression(TestCase):

    def test_rolling_regressor_basic(self):

        x = np.arange(12).reshape(6, 2)
        y = np.arange(6)

        reg = RollingRegressor(x, y)
        self.assertEqual((2, 6, 2), reg.x.shape)
        self.assertEqual((6, 1), reg.y.shape)
        res = reg.fit(5)
        np.testing.assert_array_almost_equal(
            array(
                [
                    [nan, nan],
                    [nan, nan],
                    [nan, nan],
                    [nan, nan],
                    [0, -0.5],
                    [0, -0.5],
                ]
            ),
            res.alpha,
        )
        np.testing.assert_array_almost_equal(
            array(
                [
                    [nan, nan],
                    [nan, nan],
                    [nan, nan],
                    [nan, nan],
                    [0.5, 0.5],
                    [0.5, 0.5],
                ]
            ),
            res.beta,
        )

    def test_rolling_regression(self):
        x = pd.DataFrame(np.arange(12).reshape(6, 2), index=list("ABCDEF"), columns=list("ab"), dtype=float)
        y = x.copy()
        y.iloc[3:, 0] = x.iloc[3:, 0] * 2 + 1
        y.iloc[3:, 1] = x.iloc[3:, 1] * 4 + 1

        res0 = rolling_regression(x, y, 3)
        pd.testing.assert_frame_equal(
            res0.alpha,
            pd.DataFrame(
                array([[nan, nan], [nan, nan], [0, 0], [-4.66666667, -20.1666667], [-8.16666667, -32.3333333], [1, 1]]),
                index=list("ABCDEF"),
                columns=list("ab"),
            ),
        )
        pd.testing.assert_frame_equal(
            res0.beta,
            pd.DataFrame(
                array([[nan, nan], [nan, nan], [1, 1], [2.75, 6.5], [3.25, 8], [2, 4]]),
                index=list("ABCDEF"),
                columns=list("ab"),
            ),
        )

        res1 = rolling_regression([x, x], y, 3)
        pd.testing.assert_frame_equal(res0.alpha, res1.alpha)
        pd.testing.assert_frame_equal(
            pd.concat(res1.beta, axis=1),
            pd.DataFrame(
                array(
                    [
                        [nan, nan, nan, nan],
                        [nan, nan, nan, nan],
                        [0.5, 0.5, 0.5, 0.5],
                        [1.375, 3.25, 1.375, 3.25],
                        [1.625, 4.0, 1.625, 4.0],
                        [1.0, 2.0, 1.0, 2.0],
                    ]
                ),
                index=list("ABCDEF"),
                columns=list("abab"),
            ),
        )

    def test_table_regression(self):
        x = pd.DataFrame(np.arange(12).reshape(2, 6), index=list("ab"), columns=list("ABCDEF"), dtype=float)
        y = x.copy()
        y.iloc[0] = x.iloc[0] * 2 + 1
        y.iloc[1] = x.iloc[1] * 4 + 1
        w = x.copy()
        w.iloc[:] = 1

        for _w in [None, w]:
            with self.subTest(w=_w):
                res0 = table_regression(x, y, _w, axis=0)
                pd.testing.assert_frame_equal(res0.alpha + res0.beta * x, y)

                res1 = table_regression(x, y, _w, axis=1)
                pd.testing.assert_frame_equal(x.mul(res1.beta, axis=0).add(res1.alpha, axis=0), y)

    def test_table_regression_weights(self):
        x = pd.DataFrame(np.arange(12).reshape(2, 6), index=list("ab"), columns=list("ABCDEF"), dtype=float)
        y = x.copy()
        y.iloc[0] = x.iloc[0] * 2 + 1
        y.iloc[1] = x.iloc[1] * 4 + 1
        w = x.copy()

        # 此时w不影响结果
        res0 = table_regression(x, y, w, axis=1)
        pd.testing.assert_frame_equal(x.mul(res0.beta, axis=0).add(res0.alpha, axis=0), y)


    # test rolling regressor with fit_intercept=False
    def test_rolling_regressor_no_intercept(self):
        x = gen_df()
        y = gen_df()

        res = RollingRegressor(x, y, fit_intercept=False).fit(window=5, n_jobs=1)
        print(res.alpha)
        print(res.beta)

    # 测试多元线性回归
    def test_multi_linear_regression(self):
        x = list(gen_df(10,3) for _ in range(3))
        y = gen_df((10,3))

        res = RollingRegressor(x, y, fit_intercept=False).fit(window=5, n_jobs=1)
        print(f"alpha: {res.alpha}")
        print(f"beta: {res.beta}")