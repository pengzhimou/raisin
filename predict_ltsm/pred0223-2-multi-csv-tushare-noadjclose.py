import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tushare as ts

#####数据准备开始
# ###### 读取历史股票数据
# data = pd.read_csv('yhAAPL.csv')

# # 数据预处理 - 移除Adj Close列
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'close', 'Volume']])

###### 设置Tushare token
ts.set_token('c75d66a12f099b7ced441563e83234d3b73acf437f532a6759a17f10')
pro = ts.pro_api()

# 获取数据
data = pro.daily(ts_code='000020.SZ', start_date='20220101', end_date='20240116')
print(data)



# 反转数据，将数据按时间顺序排列
data = data.iloc[::-1]

# 选择使用的列
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'vol']])

#####数据准备结束

# 构建训练集和测试集
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 3])  # 使用Close列作为预测目标
    return np.array(X), np.array(y)

time_step = 30  # 时间步长，可以根据需要调整
X, y = create_dataset(scaled_data, time_step)

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 滚动预测未来5天股票走势
pred_days = 5
predictions = []
current_data = scaled_data[-time_step:]  # 最后time_step天的数据作为初始数据

for i in range(pred_days):
    input_data = current_data[-time_step:].reshape(1, time_step, 5)  # 输入数据调整为5列
    pred = model.predict(input_data)
    predictions.append(pred[0, 0])
    # 更新current_data时，确保现在只使用5个特征
    current_data = np.append(current_data, np.array([[current_data[-1, 0], current_data[-1, 1], current_data[-1, 2], pred[0, 0], current_data[-1, 4]]]).reshape(1, 5), axis=0)

# 将预测结果反归一化
predictions = np.array(predictions).reshape(-1, 1)
# 根据新的数据结构调整反归一化逻辑
predictions = np.concatenate((np.zeros((len(predictions), 2)), predictions, predictions, np.zeros((len(predictions), 1))), axis=1)
predictions = scaler.inverse_transform(predictions)[:, 3]

print(predictions)

# 绘制历史数据和预测数据的股价走势图
dates = data['trade_date']
close_prices = data['close']


plt.figure(figsize=(12, 6))
plt.plot(dates, close_prices, label='Close Price')
plt.plot(range(len(data), len(data) + pred_days), predictions, label='Predicted Close', color='r')
plt.xlabel('Day')
plt.ylabel('close')
plt.title('Historical and Predicted Close')
plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示日期
plt.legend()
plt.grid()
plt.show()

