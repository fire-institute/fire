import pandas as pd

# for minute data bartimes
_morning = pd.date_range("2020-01-01 09:30", "2020-01-01 11:30", freq="1 min")
_afternoon = pd.date_range("2020-01-01 13:00", "2020-01-01 15:00", freq="1 min")
MIN_BARTIMES = _morning.union(_afternoon).strftime("%H:%M")