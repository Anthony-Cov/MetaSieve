
import pandas as pd
import numpy as np
from scipy.linalg import hankel
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.ensemble import RandomForestRegressor
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import functools

import matplotlib.pyplot as plt
from timeit import Timer

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


def RF_tuning(data, lag = 20, estimators = 2, bound=400, fwd=5):
    '''Metrics: x-prediction, y-real; Out - : sMAPE'''
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
        return m, d1, d2, d3, d4
    
    Xy=hankel(data)[:-lag-fwd,:lag+fwd]
    
    model = RandomForestRegressor(n_estimators=pow(2,estimators))
    smapes=[]
    for i in range(10):
        X_train, y_train = Xy[100+i:bound+fwd+i, :lag], Xy[100+i:bound+fwd+i, lag:]
        X_test, y_test = Xy[bound+fwd+i:bound+fwd+i+1, :lag], Xy[bound+fwd+i:bound+fwd+i+1, lag:]
        if i==0:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        smapes.append(Metr(y_pred[0],y_test[0])[3])
    del model
    return np.mean(smapes)



'''Brute-Force pass scheme'''
def forecast_scores(df, levels = 5):
    #output table
    list_of_columns = df.columns[1:].tolist()
    result = pd.DataFrame(columns = ['level']+list_of_columns)
    result["level"] = np.arange(levels)
    for level in range(0,levels):
        for colname in list_of_columns:
            data = df[colname]
            smape = RF_tuning(data, estimators = level,  bound = 400, fwd=5)
            result.loc[level, colname] = smape
        
    return result 

'''Sieve scheme'''
def forecast_scores_cut(df, levels = 5):
    list_of_columns = df.columns[1:].tolist()
    result = pd.DataFrame(columns = ['level']+list_of_columns)
    result["level"] = np.arange(levels)
    for level in range(0,levels):
        mse_on_level = []
        for colname in list_of_columns:
            data = df[colname]
            smape = RF_tuning(data, estimators = level,  bound = 400, fwd=5)
            result.loc[level, colname] = smape
            mse_on_level.append(smape)
        
        bad = [i for i,v in enumerate(mse_on_level) if v > np.median(mse_on_level)]
        list_of_columns = [list_of_columns[i] for i in bad]

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

data = df


'''Measure the time for the brute-force method'''
time = []
for n_data in range(2,102):
    print(n_data)
    df = data.iloc[:,0:n_data]
    t = Timer(functools.partial(forecast_scores,df))  
    time_eval = t.timeit(1)
    print(time_eval)
    time.append(time_eval)


brute = pd.DataFrame()
brute['time'] = time
brute['ncols'] = range(1,101)







'''Measure the time for the Sieve method'''
time_cut = []
for n_data in range(2,102):
    print(n_data)
    df = data.iloc[:,0:n_data]
    t = Timer(functools.partial(forecast_scores_cut,df))  
    time_eval = t.timeit(1)
    print(time_eval)
    time_cut.append(time_eval)


cut = pd.DataFrame()
cut['time'] = time_cut
cut['ncols'] = range(1,101)



'''Plot results'''
plt.figure(figsize=(8,6), dpi= 80)
plt.plot(brute['ncols'], brute['time'], label = "Brute-Force")
plt.plot(cut['ncols'], cut['time'], label = "Sieve")
plt.legend()
plt.ylabel('Calculation time (s.)',fontsize=15)
plt.xlabel('Number of series', fontsize=15)
plt.show()

'''Percentage change'''
(1-np.mean((brute['time'] - cut['time'])/ brute['time']))*100




