import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random
import functions
from error_functions import *
import LNL_model_modified
import hybrid_ann
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
import seaborn as sns
sns.set()
from matplotlib.pylab import rcParams
import glob
import warnings
warnings.filterwarnings('ignore')

##############################READING ALL DATASETS##############################

def main():   
    selected = './data/'

    dbs = glob.glob(selected + '*.txt')

    for ds, dataset in enumerate(dbs):
        #Pre processing

        arquivo = dataset
        nome_serie = dataset.split('/')[-1].split('.')[0]

        print('Dataset: ' + nome_serie)

        df = pd.read_csv(arquivo, sep = '\t', header = None)
        data = df.iloc[:, -1].values

        serie_normal = functions.normalizar_serie(data)
        max_lag = 10
        lags_acf = functions.select_lag_acf(serie_normal, max_lag)
        max_sel_lag = lags_acf[0]

        serie_lags = functions.gerar_janelas(tam_janela = max_sel_lag+1, serie = serie_normal) #Sem normalizar
        X_train, y_train, X_test, y_test = functions.split_serie_with_lags(serie = serie_lags, perc_train = 0.7)

        #utiliza apenas os lags selecionados pelo o ACF
        X_train = X_train[:, lags_acf]
        X_test = X_test[:, lags_acf]

        #----------------------------------Experiments----------------------------------#

        epochs = 50000
        n_particles = 30
        modelos = []
        MSE_teste = []
        MAE_teste = []
        MAPE_teste = []
        ARV_teste = []

        average_error = np.zeros(int(epochs/10))

        repetitions = 10
        for i in range(repetitions):
            functions.update_progress(i/ repetitions)
            model = hybrid_ann.LNL_ANN(m = len(lags_acf))
            model.fit_MPSO(X_train, y_train, n_particles, c1 = 2.0, c2 = 2.0, w = 1, maxt = epochs)
            modelos.append(model)    
            y_pred = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_pred)
            test_arv = average_relative_variance(y_test, y_pred)
        #     print(test_mse)
            MSE_teste.append(test_mse)
            MAE_teste.append(test_mae)
            MAPE_teste.append(test_mape)
            ARV_teste.append(test_arv)

            average_error += np.array(model.MSE_gBest)

        average_error /= repetitions
        np.savetxt('./results/paper/' + nome_serie + "_meanMSE.csv", average_error , delimiter="\t")
        del average_error
        functions.update_progress(1)

        #melhor modelo das N iterações
        indices = range(0, len(MSE_teste))    
        valores_ordenados, indices = zip(*sorted(zip(MSE_teste, indices)))
        best_model = modelos[indices[0]]

        previsoes = []
        for i in range(0, 10):
            model = modelos[i]
            y_pred = model.predict(X_test)
            previsoes.append(y_pred)
            

        previsoes_np = np.array(previsoes)
        previsoes_pd = pd.DataFrame(previsoes_np)
        previsoes_pd = previsoes_pd.transpose()
        target_pd = pd.DataFrame(y_test)

        metrics = pd.DataFrame(data = {'MSE': MSE_teste, 'MAE': MAE_teste, 'MAPE': MAPE_teste, 'ARV': ARV_teste})

        d = target_pd.astype(float)
        d2 = previsoes_pd.astype(float)
        metrics = metrics.astype(float)

        name_save_result = './results/paper/' + nome_serie + '_previsoes_PropostoArtigo.xlsx'
        writer = pd.ExcelWriter(name_save_result)
        d.to_excel(writer,'Target')
        d2.to_excel(writer, 'Predict')
        metrics.to_excel(writer,'Metrics test')

        writer.save()

        arquivo_salvar = './results/paper/' + nome_serie+"_PropostoArtigo.txt"
        f = open(arquivo_salvar, "a+")
        f.write('----------------- Modelo -----------------')
        f.write('\n\nSuper particle = ' + repr(best_model.weight))

        f.close()
        print('\n---------------------------------------------\n')

if __name__== "__main__":
    main()