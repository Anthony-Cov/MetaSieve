import pandas as pd
import numpy as np
from scipy.linalg import hankel
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def seconds_to_str(seconds):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%02d:%02d:%02d" % (hh, mm, ss)

'''Линейное укладывание в диапазон [0,1], возвращает коэффициенты для восстановления (max(X))!=0'''
def Norm01(x):
    mi=np.nanmin(x)
    ma=np.nanmax(np.array(x)-mi)
    if ma>0.:
        x_n=(np.array(x)-mi)/ma
        return x_n, mi, ma
    else:
        return np.zeros(len(x)), mi, ma

def LSTM_tuning(data, metric='sMAPE', lag = 20, layers = 2, bound=400, fwd=5):
    '''Metrics: x-prediction, y-real; Out - :ME, MAE, MAPE, sMAPE, RMSE'''
    def Metr(x,y):
        y=np.array(y)
        x=np.array(x)
        d=x-y
        m=np.mean(d)
        d1=np.mean(abs(d))
        if sum(y):
            d2=sum([abs(d[i]/y[i]) for i in range(len(y)) if y[i]!=0])/len([y[i]for i in range(len(y)) if y[i]!=0 ])*100
            d3=sum([abs(d[i]/(x[i]+y[i])*2) for i in range(len(y)) if (x[i]+y[i])!=0])/len([y[i] 
                                                            for i in range(len(y)) if (x[i]+y[i])!=0 ])*100
        else:
            d2,d3 = np.nan, np.nan
        d4=np.std(d)
        return m, d1, d2, d3, d4 #:ME, MAE, MAPE, sMAPE, RMSE
    
    Xy=hankel(data)[:-lag-1,:lag+fwd]
    if metric=='sMAPE': 
        mind=3
    else:
        mind=4
    model = Sequential()
    model.add(LSTM(64*layers, input_shape=(lag, 1)))
    for i in range(0,layers):
        model.add(Dense(64*layers//(i+1), activation='relu'))
    model.add(Dense(fwd))
    model.compile(loss='mean_squared_error',optimizer='adam')
    smapes=[]
    for i in range(10):
        X_train, y_train = Xy[100+i:bound+fwd+i, :lag], Xy[100+i:bound+fwd+i, lag:]
        X_test, y_test = Xy[bound+fwd+i:bound+fwd+i+1, :lag], Xy[bound+fwd+i:bound+fwd+i+1, lag:]
        if i==0:
            model.fit(X_train, y_train, epochs=100, verbose=0)
        y_pred = model.predict(X_test)
        smapes.append(Metr(y_pred[0],y_test[0])[mind])# [3] - for sMAPE, 4 -for RMSE
    return np.mean(smapes)

'''Pass through all levels with all data'''
def data_meta_learning_equal_meaniter(df, metric='sMAPE', levels = 5):
    #output table
    list_of_columns = df.columns[1:].tolist()
    result = pd.DataFrame(columns = ['level', 'time']+list_of_columns)
    #Create empty table
    result["level"] = np.arange(levels)
    for level in range(0,levels):
        s=time()
        print('level', level)
        for colname in tqdm(list_of_columns):
            data = df[colname]
            if metric=='RMSE':
                data= Norm01(data.values)[0]
            smape = LSTM_tuning(data, metric=metric, layers = 2 + level,  bound = 400, fwd=5)
            result.loc[level, colname] = smape
        result.loc[level, 'time'] = time()-s
        print('\n\tTime: '+seconds_to_str(time()-s))
    return result 

newdat=False
datafile='artdata_1000.csv'
metric='RMSE'  #sMAPE'
df=pd.read_csv(datafile)
print(datafile, 'loaded: %i series'%(len(df.columns)-1))

'''Collect quality metrics'''
result = data_meta_learning_equal_meaniter(df, metric=metric, levels = 5)
result.to_csv(datafile[:-4]+'LSTM_RMSE.csv', index=False)
print(datafile[:-4]+'LSTM_RMSE.csv saved.')
