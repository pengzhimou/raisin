import tushare as ts
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置Tushare token
ts.set_token('c75d66a12f099b7ced441563e83234d3b73acf437f532a6759a17f10')
pro = ts.pro_api()

# 获取数据
data = pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20231231')

# 反转数据，将数据按时间顺序排列
data = data.iloc[::-1]

# 选择使用的列
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'vol']])

# 提取2023年的日期和收盘价数据
dates = data['trade_date']
close_prices = data['close']

# 绘制2023年全年收盘价走势图
plt.figure(figsize=(14, 7))
plt.plot(dates, close_prices, label='Close Price')
plt.title('2023 Close Prices of Stock 000001.SZ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示日期
plt.legend()
plt.grid()
plt.show()
