import pandas as pd
import numpy as np
from scipy.linalg import hankel
from time import time
import xgboost as xgb
from tqdm import tqdm

def XGB_tuning(data, lag = 20, estimators = 2, bound=400, fwd=5):
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
    # задаём параметры
    params = {'objective': 'reg:squarederror','booster':'gblinear'}
    trees = 5+estimators*5 
    smapes=[]
    for i in range(10): 
        X_train, y_train = Xy[100+i:bound+fwd+i, :lag], Xy[100+i:bound+fwd+i, lag:]
        X_test, y_test = Xy[bound+fwd+i:bound+fwd+i+1, :lag], Xy[bound+fwd+i:bound+fwd+i+1, lag:]
        #XGBOOST
        y_pred=[]
        for j in range(fwd):
            dtrain = xgb.DMatrix(X_train, label=y_train[:, j].copy())
            dtest = xgb.DMatrix(X_test)
            bst = xgb.train(params, dtrain, num_boost_round=trees)
            # посмотрим, как модель вела себя на тренировочном отрезке ряда
            y_pred.append(bst.predict(dtest)[0])
        smapes.append(Metr(y_pred,y_test[0])[3])
    return np.mean(smapes)

'''get data from file'''
datafile='artdata_1000.csv'
df=pd.read_csv(datafile)
print(datafile, 'loaded: %i series'%(len(df.columns)-1))

levels=15
result=pd.DataFrame(columns=['level', 'time']+list(df.columns[1:]))
result['level']=np.arange(levels)

for level in range(levels):
    print('level', level)
    start_time = time()
    for colname in tqdm(df.columns[1:]):
        data = df[colname]
        smape = XGB_tuning(data, lag = 5, estimators = level,  bound = 400, fwd=5)
        result.loc[level, colname] = smape
    result.loc[level, 'time'] = time()-start_time
result.to_csv('art1000_XGB_acc_time.csv', index=False)
print('Result saved as \"art1000_XGB_acc_time.csv\"')