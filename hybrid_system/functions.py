import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def normalizar_serie(serie):
	minimo = min(serie)
	maximo = max(serie)
	y = (serie - minimo) / (maximo - minimo)
	return y

def desnormalizar(serie_atual, serie_real):
	minimo = min(serie_real)
	maximo = max(serie_real)

	serie = (serie_atual * (maximo - minimo)) + minimo

	return pd.DataFrame(serie)

def gerar_janelas(tam_janela, serie):
	# serie: vetor do tipo numpy ou lista
	tam_serie = len(serie)
	tam_janela = tam_janela +1 # Adicionado mais um ponto para retornar o target na janela

	janela = list(serie[0:0+tam_janela]) #primeira janela p criar o objeto np
	janelas_np = np.array(np.transpose(janela))    
	   
	for i in range(1, tam_serie-tam_janela):  #começa do 1 
	    janela = list(serie[i:i+tam_janela])
	    j_np = np.array(np.transpose(janela))        
	    
	    janelas_np = np.vstack((janelas_np, j_np))
	    

	return janelas_np
	
def select_lag_acf(serie, max_lag):
    from statsmodels.tsa.stattools import acf
    x = serie[0: max_lag+1]
    
    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05, fft=False,
                             unbiased=False)
    
    
    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []
    
    for i in range(1, max_lag+1):

        
        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0
    
    #caso nenhum lag seja selecionado, essa atividade de seleção para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)

    #inverte o valor dos lags para usar na lista de dados
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]



    return lags_selecionados	

	
	

def split_serie_with_lags(serie, perc_train, perc_val = 0):
	#faz corte na serie com as janelas já formadas 
	x_date = serie[:, 0:-1]
	y_date = serie[:, -1]        
	   
	train_size = np.fix(len(serie) *perc_train)
	train_size = train_size.astype(int)

	if perc_val > 0:        
	    val_size = np.fix(len(serie) *perc_val).astype(int)  
	    x_train = x_date[0:train_size,:]
	    y_train = y_date[0:train_size]
	    print("Particao de Treinamento:", 0, train_size  )
	    
	    x_val = x_date[train_size:train_size+val_size,:]
	    y_val = y_date[train_size:train_size+val_size]
	    
	    print("Particao de Validacao:",train_size, train_size+val_size)
	    
	    x_test = x_date[(train_size+val_size):-1,:]
	    y_test = y_date[(train_size+val_size):-1]
	    
	    print("Particao de Teste:", train_size+val_size, len(y_date))
	    
	    return x_train, y_train, x_test, y_test, x_val, y_val
	    
	else:
	    
	    x_train = x_date[0:train_size,:]
	    y_train = y_date[0:train_size]

	    x_test = x_date[train_size:-1,:]
	    y_test = y_date[train_size:-1]

	    return x_train, y_train, x_test, y_test

def _error(actual: np.ndarray, predicted: np.ndarray):
	return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
	EPSILON = 1e-10
	return _error(actual, predicted) / (actual + EPSILON)

def RMSE(y,output):
  	return np.sqrt(mean_squared_error(y, output))

def MAPE(actual: np.ndarray, predicted: np.ndarray):
	return np.mean(np.abs(_percentage_error(actual, predicted)))

def MdAPE(actual: np.ndarray, predicted: np.ndarray):
	return np.median(np.abs(_percentage_error(actual, predicted)))

def DA(actual: np.ndarray, predicted: np.ndarray):
	sign_vector = np.sign(np.multiply((actual[1:] - actual[:-1]), (predicted[1:] - actual[:-1])))
	for i, element in enumerate(sign_vector):
		if element == -1:
			sign_vector[i] = 0
	return np.mean(sign_vector)

def sigmoid(x, derivative=False):
    try:
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
    except:
        if x > 0:
            sigm = 1.0
        elif x < 0:
            sigm = 0.0
    return sigm

def update_progress(progress, load = 'Progress'):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))

#     clear_output(wait = True)
    text = load + ": [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text, end = '\r')