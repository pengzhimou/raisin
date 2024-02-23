import tushare as ts
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 设置Tushare token
ts.set_token('c75d66a12f099b7ced441563e83234d3b73acf437f532a6759a17f10')
pro = ts.pro_api()

# 获取数据
data = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20241115')

# 选择使用的列
features = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
data = data[features]

# 数据预处理：归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 准备数据集的函数
def prepare_data(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future +1):
        X.append(data[i - n_past:i, 0:data.shape[1]])
        y.append(data[i + n_future - 1:i + n_future, 3])  # 使用'close'列作为预测目标
    return np.array(X), np.array(y)

n_past = 60  # 使用过去60天的数据
n_future = 1  # 预测未来1天
X, y = prepare_data(data_scaled, n_past, n_future)

# 创建LSTM模型
model = Sequential([
    LSTM(100, activation='relu', input_shape=(n_past, len(features))),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# 滚动预测未来7天的走势
future_steps = 7
predictions = []

last_sequence = data_scaled[-n_past:]
current_sequence = last_sequence.reshape((1, n_past, len(features)))

for i in range(future_steps):
    predicted_value = model.predict(current_sequence)[0]
    predictions.append(predicted_value)
    
    # 更新当前序列以包括新预测
    new_row = np.zeros((1, len(features)))  # 创建一个形状匹配特征的全零数组
    new_row[0, 3] = predicted_value  # 假设'close'是第四列，索引为3
    current_sequence = np.append(current_sequence[:, 1:, :], [new_row], axis=1)

# 反转归一化以获取实际预测值
# 注意：我们只对'close'列进行了预测，因此只需要反转这一列的归一化
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
predictions = close_scaler.inverse_transform(np.array(predictions)[:,0].reshape(-1, 1))

print(predictions)

import matplotlib.pyplot as plt

# 假设你已经有了整个时间序列的历史收盘价数据
# 假设这个历史数据存储在名为data的Pandas DataFrame中的'close'列
# 同样，假设predictions变量包含了你的模型预测的未来几天的收盘价

# 获取历史收盘价数据
historical_prices = data['close'].values

# 为了在图表上区分历史数据和预测数据，我们需要知道预测数据的开始位置
# 这个位置等于历史数据的长度
start_of_forecast = len(historical_prices)

# 生成一个包含所有日期的列表，这里我们简化处理，使用整数序列来代表时间
# 假设历史数据加上预测数据总共有N天
N = len(historical_prices) + len(predictions)
time_series = list(range(N))

# 绘制历史收盘价和预测收盘价
plt.figure(figsize=(12, 7))
plt.plot(time_series[:start_of_forecast], historical_prices, label='Historical Close Price', color='blue')
plt.plot(time_series[start_of_forecast-1:N], np.append(historical_prices[-1], predictions.flatten()), label='Predicted Close Price', color='red', linestyle='--')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('Stock Price Prediction with Historical Data')
plt.xlabel('Time (days)')
plt.ylabel('Price')

# 显示图形
plt.show()
