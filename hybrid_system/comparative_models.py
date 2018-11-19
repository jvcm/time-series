import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy import *
from sklearn.neural_network import MLPRegressor
import functions
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#Metodo convencional linear
def sarima (time_serie, order, seasonal_order):
    prev = []
    history = [x for x in time_serie]
    for i in range(len(time_serie)):
        model=sm.tsa.statespace.SARIMAX(endog=history, order=order,seasonal_order = seasonal_order).fit()
        forecast = model.forecast()[0]
        prev.append(forecast)
        obs = time_serie[i]
        history.append(obs)
    return prev

#METODO FFANN - MLP
#hidden_layer - tupla com a quantidade de camadas ocultas. Cada elemento da tupla equivale a uma camada de
#neuronios
def FFANN (hidden_layer, input_train_data, output_train_data, input_test):
    ffann = MLPRegressor(hidden_layer_sizes=(10,),activation= 'logistic',solver='sgd', max_iter=1000)
    ffann.fit(input_train_data,output_train_data)
    return ffann.predict(input_test)


#METODO WMES
#order - corresponde a ordem em que se deseja verificar a sazonalidade
#n_point - corresponde ao numero de pontos que se deseja a previsao
def WMES(time_serie, order, n_point):
    fit2 = ExponentialSmoothing(time_serie, seasonal_periods=order, trend='add', seasonal='mul').fit(use_boxcox=True)
    result = fit2.forecast(n_point)
    return result

#main para teste
def main():
    df = pd.read_csv('beer.csv')
    data = df.iloc[:, -1].values

    serie_normal = functions.normalizar_serie(data)
    serie_lags = functions.gerar_janelas(tam_janela=5, serie=serie_normal)
    X_train, y_train, X_test, y_test = functions.split_serie_with_lags(serie=serie_lags, perc_train=0.8, perc_val=0)

    #UTILIZANDO O MODELO SARIMA
    #modelo = sarima(data,(0,1,1),(0,1,1,4))
    #print("modelo", modelo)
    #plt.plot(modelo, label='SARIMA')
    #plt.legend(loc='best')
    #plt.show()

    #UTILIZANDO O MODELO FFANN
    #resultado = FFANN((2,),X_train, y_train, X_test)
    #plt.plot(resultado, label='MLP')
    #plt.legend(loc='best')
    #plt.show()

    #UTILIZANDO O MODELO WMES
    #resultado = WMES(data, 4,24)
    #plt.plot(resultado, label='WMES')
    #plt.legend(loc='best')
    #plt.show()

if __name__ == '__main__':
    main()