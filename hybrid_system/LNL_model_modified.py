import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random
import functions

class LNL_ANN:
	def __init__(self, m = 5, z = 2): # Rede da série (m)/ Rede do ruído (z)
		self.m = m
		self.z = z
		self.k = (3*m + 4) + (3*z + 4) + 3 # Série + Ruído + Combinação
		self.weight = np.random.rand(self.k) # Super partícula
		return

	def forward_series(self, weight, X):
		m = int((len(weight) - 4)/3)
		w1 = weight[:m]
		b1 = weight[m]
		w2 = weight[m+1:2*m+1]
		b2 = weight[2*m+1:3*m+1]
		w3 = weight[-3:-1]
		b3 = weight[-1]
		#Calculate Net1 and Net2
		net1 = np.dot(X, w1) + b1
		net2 = np.prod(X*w2 + b2)
		# print(net1, net2)
		#Activation Functions
		f1 = net1
		# print(net2)
		try:
			exp = math.exp(-net2)
			div  = (1.0 + exp)
			f2 = 1.0 / div
		except:
			if net2<0:
				f2 = 0
			elif net2>0:
				f2 = 1		
		# print(f2)
		#Calculate Net3
		net3 = w3[0]*f1 + w3[1]*f2 + b3
		# print(net3)
		return net3
	
	def fit_MPSO(self, X, y, d = 30, c1i = 2.0, c1f = 3.0, c2i = 2.0, c2f = 3.0,
		w1 = 0.5, w2 = 1.0, maxt = 1000):

		particles = np.random.rand(d, self.k)
		velocity = np.random.uniform(low = -1.0, high = 1.0, size = (d, self.k))
		pBest = particles[:]
		gBest = particles[0]
		best_fitness = np.full(d, np.inf)

		for t in range(maxt):
			fitness = np.zeros(d)
			for i, p in enumerate(particles):
# 1st Layer: Series Forecasting
				predict = np.zeros(len(y))
				for j, x in enumerate(X):
					predict[j] = self.forward_series(p[: 3*self.m+4], x)

				# Residual data set
				resid_train = functions._error(y, predict)
				resid_train_lags = functions.gerar_janelas(tam_janela = self.z - 1, serie = resid_train)
				X_error_train = resid_train_lags

# 2nd Layer: Residual Forecasting
		
				predict_res = np.zeros(len(X_error_train))
				for j, x in enumerate(X_error_train):
					predict_res[j] = self.forward_series(p[3*self.m+4 : 3*self.m+4 + 3*self.z+4], x)

		# Super particle
# 3rd Layer: Combination
				X_comb = np.hstack((predict[self.z:].reshape(-1,1), predict_res.reshape(-1,1)))
				X_comb = np.hstack((X_comb, np.ones(len(X_comb)).reshape(-1,1)))

				output = np.zeros(len(X_error_train))
				for j, x in enumerate(X_comb):
					output[j] = np.dot(p[-3:], x)
				fitness[i] = mean_squared_error(y[self.z:], output)

				if fitness[i] < best_fitness[i]:
					pBest[i] = p[:]
					best_fitness[i] = fitness[i]
			gBest = particles[np.argmin(fitness)]
			bad_index = np.argmax(fitness)
			c1 = ((c1f - c1i)*t/maxt) +c1i
			c2 = ((c2f - c2i)*t/maxt) +c2i
			w = w1 + (w2 - w1)*((maxt - t)/maxt)
			for i, p in enumerate(particles):
				if i == bad_index:
					velocity[i] = np.random.uniform(low = -1.0, high = 1.0, size = self.k)
					particles[i] = np.random.rand(self.k)
					pBest[i] = particles[i]
				else:
					velocity[i] = w*velocity[i] + c1*random.uniform(0, 1)*(pBest[i] - p) + c2*random.uniform(0, 1)*(gBest - p)
					particles[i] = p +velocity[i]
		
		self.weight = gBest[:]
		return


	def predict(self, X_test):
		series = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			series[i] = self.forward_series(self.weight[:3*self.m+4], x)
		y_test = X_test[1:,-1]
		# print('X_test:%f - Predict:%f - y_test:%f'%(len(X_test), len(series), len(y_test)))
		resid = functions._error(y_test, series[1:])
		resid_window = functions.gerar_janelas(tam_janela = self.z-1, serie = resid)
		# print('resid_window:%f '%(len(resid_window)))
		pred_resid = np.zeros(len(resid_window))
		for i, x in enumerate(resid_window):
			pred_resid[i] = self.forward_series(self.weight[3*self.m+4 : 3*self.m+4 + 3*self.z+4], x)
		X_comb = np.hstack((series[self.z + 1:].reshape(-1,1), pred_resid.reshape(-1,1)))
		X_comb = np.hstack((X_comb, np.ones(len(X_comb)).reshape(-1,1)))
		# print(len(X_comb))
		predict = np.zeros(len(X_comb))
		for i, x in enumerate(X_comb):
			predict[i] = np.dot(self.weight[-3:], x)
		return predict


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('beer.csv')
data = df.iloc[:, -1].values

ent = 8

serie_normal = functions.normalizar_serie(data)
serie_lags = functions.gerar_janelas(tam_janela = ent, serie = serie_normal)
X_train, y_train, X_test, y_test = functions.split_serie_with_lags(serie = serie_lags, perc_train = 0.86, perc_val = 0)

model = LNL_ANN(m = ent, z = 2)
model.fit_MPSO(X_train, y_train, d = 70, c1i = 1.0, c1f = 2.0, c2i = 1.0, c2f = 2.0, w1 = 0.2, w2 = 0.7, maxt = 4000)
pred = model.predict(X_test)
pred = functions.desnormalizar(pred, data)
print('y_test:',data[-len(pred):],'\nprediction:',pred.values.reshape(1,-1))
plt.plot(pred, label='Prediction')
plt.plot(data[-len(pred):], label = 'Original')
plt.legend()
plt.show()