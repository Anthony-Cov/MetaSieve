import pandas as pd
import numpy as np
from scipy.linalg import hankel
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

def seconds_to_str(seconds):
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%02d:%02d:%02d" % (hh, mm, ss)

# n - series length
def trans(n):   # data drift
    x=np.linspace(-n*0.009, n*0.009, n)
    y=1-1/(1+np.exp(x))
    return y

def sin(n):
    x=np.linspace(-n*0.15, n*0.15, n)
    y = (np.sin(x) + 1) / 2
    return y

def period(n):  # seasonality
    x=np.linspace(0,32*np.pi, n)
    y=np.abs(np.sin(x))
    return y
def noise(n):   # white noise
    y=np.random.randn(n)
    y=(y-min(y))/(max(y)-min(y)) 
    return y
def rndwalk(n): # random walk
    y=[0.]
    for i in range(n-1):
        k=np.random.rand()
        sign = 1. if np.random.randint(2) else -1. #x(t+1)=x(t)+-1
        y.append(y[-1]+sign*k)
    y=np.array(y)
    y=(y-min(y))/(max(y)-min(y)) # rescale [0..1]
    return y
def compose(n,kt,kp,kn,kr): # All together in proportion
    y=kt*sin(n) + kp * period(n) + kn * noise(n) + kr * rndwalk(n)
    y=(y-min(y))/(max(y)-min(y))
    return y

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
        i += 1
        model.add(Dense(64*layers/(i-0.5), activation='relu'))

    model.add(Dense(fwd))
    model.compile(loss='mean_squared_error',optimizer='adam')
    smapes=[]
    for i in range(15):
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
    result = pd.DataFrame(columns = ['level']+list_of_columns)
    #Create empty table
    result["level"] = np.arange(levels)
    s=time()
    for level in range(0,levels):
        print('level', level)
        for colname in list_of_columns: 
            data = df[colname]
            print(colname, end=' ') #progress bar
            smape = LSTM_tuning(data, metric=metric, layers = 2 + level,  bound = 400, fwd=5)
            result.loc[level, colname] = smape
        print('\n\tTime: '+seconds_to_str(time()-s))
    return result 

newdat=True
datafile='data.csv'
metric='sMAPE'
if newdat:
    '''create random data:'''
    n=500
    df=pd.DataFrame()
    df['t']=np.arange(n)
    for i, alpha in enumerate(np.linspace(0,1,350)): 
        y=compose(n,0,1-alpha,0,alpha)
        df=pd.merge(df, pd.DataFrame({'t':np.arange(n),str(i).zfill(3):y}), on='t', how='inner')
    print('%i artificial series created'%(i+1))
    df.to_csv(datafile, index=False)
else:
    '''get data from file'''
    df=pd.read_csv(datafile)
    print(datafile, 'loaded: %i series'%(len(df.columns)-1))

'''Collect quality metrics'''
result = data_meta_learning_equal_meaniter(df, metric=metric, levels = 5)
result.to_csv('res_'+datafile+'.csv', index=False)
print('res_'+datafile+'.csv saved.')
