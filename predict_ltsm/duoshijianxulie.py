import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# from keras.wrappers.scikit_learn import KerasRegressor
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

df=pd.read_csv("train.csv",parse_dates=["Date"],index_col=[0])
# df.shape # (5203, 5)



test_split=round(len(df)*0.20)
df_for_training=df[:-1041]
df_for_testing=df[-1041:]
print(df_for_training.shape)
print(df_for_testing.shape)
# (4162, 5)
# (1041, 5)

scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)

# df_for_training_scaled.shape # (4162, 5)
# df_for_testing_scaled.shape # (1041, 5)



def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)        

trainX,trainY=createXY(df_for_training_scaled,30)
# trainX.shape # (4132, 30, 5)

testX,testY=createXY(df_for_testing_scaled,30)
# trainX[0]
# array([[0.85398707, 0.86281807, 0.85292546, 0.8403402 , 0.82180889],
#        [0.85533406, 0.85473269, 0.82623316, 0.8122593 , 0.79289309],
#        [0.82155169, 0.84260459, 0.81422168, 0.80701755, 0.78749611],
#        [0.81918098, 0.84303579, 0.82319031, 0.8470261 , 0.8286929 ],
#        [0.86206895, 0.85769729, 0.84753366, 0.84124952, 0.8227448 ],
#        [0.85668106, 0.85295391, 0.85479397, 0.84659822, 0.82825271],
#        [0.85129307, 0.85661925, 0.85372629, 0.84766796, 0.82935384],
#        [0.8540948 , 0.88249248, 0.85799703, 0.88125807, 0.86394198],
#        [0.88577588, 0.88227684, 0.88255396, 0.87590928, 0.85843431],
#        [0.88189657, 0.87602422, 0.87016871, 0.86200256, 0.84411479],
#        [0.88362063, 0.88357053, 0.87892376, 0.86606763, 0.84829995],
#        [0.87047413, 0.86200956, 0.84390347, 0.8344031 , 0.81569511],
#        [0.83857758, 0.87645542, 0.84966903, 0.87398378, 0.85645156],
#        [0.88168102, 0.88012075, 0.88105913, 0.86649551, 0.84874037],
#        [0.87090514, 0.86287196, 0.85949177, 0.84724008, 0.82891342],
#        [0.85237065, 0.88249248, 0.8601324 , 0.88403941, 0.86680588],
#        [0.85668106, 0.86589053, 0.86248134, 0.8630723 , 0.84521599],
#        [0.87176726, 0.8870203 , 0.88191333, 0.87783486, 0.86041721],
#        [0.88254305, 0.89003888, 0.88298101, 0.86949083, 0.85182511],
#        [0.875     , 0.86955587, 0.85821051, 0.86521178, 0.84741896],
#        [0.85775864, 0.85877534, 0.83600253, 0.84552848, 0.82715127],
#        [0.86745685, 0.88055195, 0.86120008, 0.88403941, 0.86680588],
#        [0.87780172, 0.88033639, 0.87828313, 0.88446729, 0.86724607],
#        [0.88900862, 0.88551106, 0.84838778, 0.85237489, 0.83420073],
#        [0.83512929, 0.83872361, 0.83365367, 0.83975181, 0.82120263],
#        [0.83189656, 0.82988361, 0.82532568, 0.81108261, 0.79168158],
#        [0.81896551, 0.82341526, 0.82703399, 0.82199401, 0.80649534],
#        [0.85129307, 0.85015098, 0.84240872, 0.8292683 , 0.81401255],
#        [0.83448273, 0.84282023, 0.84561178, 0.84124952, 0.82639405],
#        [0.84913791, 0.84497632, 0.83557547, 0.83889605, 0.82396218]])

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
# trainX Shape--  (4132, 30, 5)
# trainY Shape--  (4132,)

print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)
# testX Shape--  (1011, 30, 5)
# testY Shape--  (1011,)

def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)



grid_search = grid_search.fit(trainX,trainY)


grid_search.best_params_
my_model=grid_search.best_estimator_.model

prediction=my_model.predict(testX)













