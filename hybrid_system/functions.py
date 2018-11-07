import numpy as np

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