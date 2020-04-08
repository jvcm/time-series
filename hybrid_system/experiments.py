import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random
import functions
import hybrid_ann
import matplotlib.pyplot as plt
from RBF import RBF
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from matplotlib.pylab import rcParams
from progress.bar import IncrementalBar


### Dataset ###

name = 'Lynx'
df = pd.read_csv('./data/' + name + '.csv')
data = df.iloc[:, -1].values
data = np.log(data)

if name not in os.listdir('./results/'):
    os.mkdir('./results/' + name) # Create directory for current dataset

### Set parameters according to dataset ###

ent = 4 # Janela do modelo normal
n_exp = 40 # Numero de experimentos
tam_jan = 2 # Janela do ruido
print('Data set: ' + name)
bar = IncrementalBar('Countdown', max = n_exp)

# Sarima
order = (2,0,0)

#WMES
s_periods= 4

serie_normal = functions.normalizar_serie(data)
serie_lags = functions.gerar_janelas(tam_janela = ent, serie = serie_normal)
X_train, y_train, X_test, y_test = functions.split_serie_with_lags(serie = serie_lags, perc_train = 0.86, perc_val = 0)

# Predictions dataframe #

predictions_ffann = pd.DataFrame()
predictions_rbf = pd.DataFrame()
predictions_lnl = pd.DataFrame()
predictions_model1 = pd.DataFrame()
predictions_model2 = pd.DataFrame()
predictions_other = pd.DataFrame()

#### Experiments ####

for i in range(n_exp):
    bar.next()
    
    ############################## Hybrid LNL ANN (Original) ##############################

    model_lnl = hybrid_ann.LNL_ANN(m = ent)
    model_lnl.fit_MPSO(X = X_train, y = y_train, d= 40, c1i = 1.0, c1f = 2.0, c2i = 1.0, c2f = 2.0, w1 = 0.5, w2 = 1.0, maxt = 6000)
    y_lnl = model_lnl.predict(X_test)
        
    predictions_lnl['LNL-ANN_' + str(i + 1)] = y_lnl[:]
        
        
    ############################## Model 1 ##############################
    
    model = hybrid_ann.LNL_ANN(m = ent)
    model.fit_MPSO(X = X_train, y = y_train, d= 40, c1i = 1.0, c1f = 2.0, c2i = 1.0, c2f = 2.0, w1 = 0.5, w2 = 1.0, maxt = 5000)
    y_pred = model.predict(X_test)
    
    y_train_pred = model.predict(X_train)
    resid_train = functions._error(y_train, y_train_pred)
    resid_test = functions._error(y_test, y_pred)

    resid_train_lags = functions.gerar_janelas(tam_janela = tam_jan, serie = resid_train)
    resid_test_lags = functions.gerar_janelas(tam_janela = tam_jan, serie = resid_test)

    ### Training/test
    X_error_train = resid_train_lags[:, :-1]
    y_error_train = resid_train_lags[:, -1]
    
    X_error_test = resid_test_lags[:, :-1]
    y_error_test = resid_test_lags[:, -1]
    
    model_resid = hybrid_ann.LNL_ANN(m = tam_jan)
    model_resid.fit_MPSO(X = X_error_train, y = y_error_train, d= 40, c1i = 1.0, c1f = 2.0, c2i = 1.0, c2f = 2.0, w1 = 0.2, w2 = 0.8, maxt = 1000)
    
    resid_pred = model_resid.predict(X_error_test) + y_pred[tam_jan+1:]
        
    predictions_model1['Model1_' + str(i + 1)] = resid_pred[:]
        
    ############################## Model 2 ##############################
    
    X_train_e = np.hstack((X_train[1:, :], resid_train[:-1].reshape(-1, 1)))
    X_test_e = np.hstack((X_test[1:, :], resid_test[:-1].reshape(-1,1)))
    
    model2 = hybrid_ann.Hybrid_ANN(model.weight)
    model2.fit_MPSO(X = X_train_e, y = y_train[1:], d= 40, c1i = 1.0, c1f = 2.0, c2i = 1.0, c2f = 2.0, w1 = 0.2, w2 = 0.8, maxt = 1000)
    
    y_pred2 = model2.predict(X_test_e)
        
    predictions_model2['Model2_' + str(i + 1)] = y_pred2[:]
        
    ############################## RBF ##############################
    
    rbf = RBF(ent, 20, 1)
    rbf.train(X_train, y_train)
    z = rbf.test(X_test)
    # z = functions.desnormalizar(z, data).iloc[:,0].values
    
    predictions_rbf['RBF_' + str(i + 1)] = z[:]
    
    ############################## FFANN ##############################
    
    ffann = MLPRegressor(hidden_layer_sizes=(20,), max_iter=2000)
    ffann.fit(X_train, y_train)
    ffann_pred = ffann.predict(X_test)
    
    predictions_ffann['FFANN_' + str(i + 1)] = ffann_pred[:]

bar.finish()

############################## SARIMA ##############################

ref = len(data) - len(y_test)
historico = [x for x in serie_normal[:ref]]
previsoes = []
for i in range(len(y_test)):
    modelo = sm.tsa.statespace.SARIMAX(historico, order= order,seasonal_order=(0,1,1,4)).fit()
    prev = modelo.forecast()[0]
    previsoes.append(prev)
    obs = y_test[i]
    historico.append(obs)
sarima = np.array(previsoes)

############################## WMES ##############################

# fit2 = ExponentialSmoothing(serie_normal[:ref-2], seasonal_periods = s_periods, trend='add', seasonal='mul').fit(use_boxcox=True)
# wmes = fit2.forecast(len(y_test))

######################################################################
######################################################################
######################################################################


############################## Results ##############################

predictions_other['Y_true'] = y_test[:]
predictions_other['SARIMA'] = sarima[:]
# predictions_other['WMES'] = wmes[:]

predictions_rbf.to_csv('./results/' + name + '/rbf.csv', index= False)
predictions_ffann.to_csv('./results/' + name + '/ffann.csv', index= False)
predictions_lnl.to_csv('./results/' + name + '/lnl.csv', index= False)
predictions_model1.to_csv('./results/' + name + '/model1.csv', index= False)
predictions_model2.to_csv('./results/' + name + '/model2.csv', index= False)
predictions_other.to_csv('./results/' + name + '/other.csv', index= False)
