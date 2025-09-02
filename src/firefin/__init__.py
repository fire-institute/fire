# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

from .compute.window import *
from .data.gateway import fetch_data
from .core.eva_utils import compute_forward_returns, compute_ic, compute_quantile_returns
from .evaluation.industry.evaluator import Evaluator
from .core.plot.plots import plt_ic, plt_quantile_cumulated_end_returns, plt_quantile_cumulative_returns
