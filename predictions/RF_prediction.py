import pandas as pd
import numpy as np
from scipy.linalg import hankel
from time import time
from sklearn.ensemble import RandomForestRegressor

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
    for i in range(15):
        X_train, y_train = Xy[100+i:bound+fwd+i, :lag], Xy[100+i:bound+fwd+i, lag:]
        X_test, y_test = Xy[bound+fwd+i:bound+fwd+i+1, :lag], Xy[bound+fwd+i:bound+fwd+i+1, lag:]
        if i==0:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        smapes.append(Metr(y_pred[0],y_test[0])[3])
    del model
    return np.mean(smapes)

'''get data from file'''
datafile='data500.csv'
df=pd.read_csv(datafile)
print(datafile, 'loaded: %i series'%(len(df.columns)-1))

levels=5
result=pd.DataFrame(columns=['level', 'time']+list(df.columns[1:]))
result['level']=np.arange(levels)

for level in range(levels):
    print('Level:', level)
    start_time = time()
    for colname in df.columns[1:]:
        print('*', end='')
        data = df[colname]
        smape = RF_tuning(data, estimators = level,  bound = 400, fwd=5)
        result.loc[level, colname] = smape
    result.loc[level, 'time'] = time()-start_time
    print('')
result.to_csv('data500RF_acc_time.csv', index=False)

