import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random
import functions
from sklearn.linear_model import Perceptron

class LNL_ANN:
	def __init__(self, m = 5, z = 2): # Rede da série (m)/ Rede do ruído (z)
		self.m = m
		self.z = z
		self.k = (3*m + 4) + (3*z + 4) + 3 # Série + Ruído + Combinação
		self.weight = np.random.rand(self.k) # Super partícula
		return

	def forward_series(self, weight, X):
		w1 = weight[: self.m]
		b1 = weight[self.m]
		w2 = weight[self.m+1: 2*self.m+1]
		b2 = weight[2*self.m+1: 3*self.m+1]
		w3 = weight[3*self.m+1: 3*self.m+3]
		b3 = weight[3*self.m+3]
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

	def _fit(self, X, y, d = 30, c1i = 2.0, c1f = 3.0, c2i = 2.0, c2f = 3.0,
		w1 = 0.1, w2 = 1.0, maxt = 1000):
		m = X.shape[1]
		k = 3*m + 4
		#MPSO Algorythm

		particles = np.random.rand(d, k)
		velocity = np.random.uniform(low = -1.0, high = 1.0, size = (d, k))
		pBest = particles[:]
		gBest = particles[0]
		best_fitness = np.full(d, np.inf)

		for t in range(maxt):
			fitness = np.zeros(d)
			for i, p in enumerate(particles):
				output = np.zeros(len(y))
				for j, x in enumerate(X):
					output[j] = self.forward_series(p, x)
				fitness[i] = mean_squared_error(y, output)
				if fitness[i] < best_fitness[i]:
					pBest[i] = p[:]
					best_fitness[i] = fitness[i]
			# print(best_fitness)
			gBest = particles[np.argmin(fitness)]
			bad_index = np.argmax(fitness)
			c1 = ((c1f - c1i)*t/maxt) +c1i
			c2 = ((c2f - c2i)*t/maxt) +c2i
			w = w1 + (w2 - w1)*((maxt - t)/maxt)
			for i, p in enumerate(particles):
				if i == bad_index:
					velocity[i] = np.random.uniform(low = -1.0, high = 1.0, size = k)
					particles[i] = np.random.rand(k)
					pBest[i] = particles[i]
				else:
					velocity[i] = w*velocity[i] + c1*random.uniform(0, 1)*(pBest[i] - p) + c2*random.uniform(0, 1)*(gBest - p)
					particles[i] = p +velocity[i]

		#Optimal solution. The L&NL-ANN weights will be gBest
		return gBest
	
	def fit_MPSO(self, X, y, d = 30, c1i = 2.0, c1f = 3.0, c2i = 2.0, c2f = 3.0,
		w1 = 0.5, w2 = 1.0, maxt = 1000):
# 1st Layer: Series Forecasting
		weight_series = self._fit(X, y, d = d, c1i = c1i, c1f = c1f, c2i = c2i, c2f = c2f,
		w1 = w1, w2 = w2, maxt = maxt)

		predict = np.zeros(len(y))
		for i, x in enumerate(X):
			predict[i] = self.forward_series(weight_series, x)

		# Residual data set
		resid_train = functions._error(y, predict)
		resid_train_lags = functions.gerar_janelas(tam_janela = self.z, serie = resid_train)
		X_error_train = resid_train_lags[:, :-1]
		y_error_train = resid_train_lags[:, -1]

# 2nd Layer: Residual Forecasting
		weight_residual = self._fit(X_error_train, y_error_train, d = d, c1i=c1i-1, c1f=c1f-1, c2i=c2i-1, c2f=c2f-1, w1= w1-0.2, w2=w2-0.2, maxt = maxt)
		
		predict_res = np.zeros(len(y_error_train))
		for i, x in enumerate(X_error_train):
			predict_res[i] = self.forward_series(weight_residual, x)

		# Super particle
		self.weight[:self.k - 3] = np.append(weight_series, weight_residual)
# 3rd Layer: Combination
		X_comb = np.hstack((predict[self.z + 1:].reshape(-1,1), predict_res.reshape(-1,1)))
		X_comb = np.hstack((X_comb, np.ones(len(X_comb)).reshape(-1,1)))

		particles = np.random.rand(d, 3)
		velocity = np.random.uniform(low = -1.0, high = 1.0, size = (d, 3))
		pBest = particles[:]
		gBest = particles[0]
		best_fitness = np.full(3, np.inf)

		for t in range(maxt):
			fitness = np.zeros(d)
			for i, p in enumerate(particles):
				output = np.zeros(len(y_error_train))
				for j, x in enumerate(X_comb):
					output[j] = np.dot(p, x)
				fitness[i] = mean_squared_error(y[self.z+1:], output)
				if fitness[i] < best_fitness[i]:
					pBest[i] = p[:]
					best_fitness[i] = fitness[i]
			# print(best_fitness)
			gBest = particles[np.argmin(fitness)]
			bad_index = np.argmax(fitness)
			c1 = ((c1f - c1i)*t/maxt) +c1i
			c2 = ((c2f - c2i)*t/maxt) +c2i
			w = w1 + (w2 - w1)*((maxt - t)/maxt)
			for i, p in enumerate(particles):
				if i == bad_index:
					velocity[i] = np.random.uniform(low = -1.0, high = 1.0, size = k)
					particles[i] = np.random.rand(k)
					pBest[i] = particles[i]
				else:
					velocity[i] = w*velocity[i] + c1*random.uniform(0, 1)*(pBest[i] - p) + c2*random.uniform(0, 1)*(gBest - p)
					particles[i] = p +velocity[i]
		weight_combination = gBest[:]
		self.weight[-3:] = weight_combination
		return


	def predict(self, X_test):
		predict = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			predict[i] = self.forward(self.weight, x)
		return predict
