# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

from firefin.data.gateway import fetch_data, DATA_MAPS

print(DATA_MAPS)

data = fetch_data(["open", "TradingValue","test_no_data"])

print(data["open"])
print(data["test_no_data"])
print(data["cn_bond_2y"])