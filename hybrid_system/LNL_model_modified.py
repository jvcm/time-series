import numpy as np
from error_functions import mean_squared_error
import math
import random
import functions

# Lasciate ogni speranza, voi ch'entrate!

class LNL_ANN:
	def __init__(self, m = 5, z = 2): # Series ANN (m)/ Residual ANN (z)
		self.m = m
		self.z = z
		self.k = (3*m + 4) + (3*z + 4) + 3 + 1# Series + Residual + Combination (MLP) + binary 
		self.weight = np.random.rand(self.k) # Super particle
		self._final = None
		return

	def forward_series(self, weight, X):
		m = int((len(weight) - 4)/3)
		w1 = weight[:m]
		b1 = weight[m]
		w2 = weight[m+1:2*m+1]
		b2 = weight[2*m+1:3*m+1]
		w3 = weight[3*m+1:3*m+3]
		b3 = weight[3*m+3]
		
		#Activation Functions
		net1 = np.dot(X, w1) + b1
		f1 = net1
		net2 = np.prod(X*w2 + b2)
		f2 = functions.sigmoid(net2)
		
		#Calculate Net3
		net3 = w3[0]*f1 + w3[1]*f2 + b3
		return net3
		
		
	def forward_with_decision(self, weight, X):
		#corrigir os indices da particula
		
# 		print('tam part', len(weight))
		
		m = int((len(weight) - 4)/3)
		w1 = weight[:m]
		b1 = weight[m]
		w2 = weight[m+1:2*m+1]
		b2 = weight[2*m+1:3*m+1]
		w3 = weight[3*m+1:3*m+3]
		b3 = weight[3*m+3]

		b4 = weight[-1] 

		
		bit_01 = 0
		bit_02 = 0
		
		if b4 < 0.3:
			bit_01 = 1
		elif b4>= 0.3 and b4 <0.6:
			bit_02 = 1
		else:
			bit_01 = 1
			bit_02 = 1
			
		
# 		print('b4', b4)
	
					
			
# 		print('-------------------------------------------------')
# 		print('bit_01', bit_01)
# 		print('bit_02', bit_02)
		
		net1 = np.dot(X, w1) + b1
		f1 = net1 * bit_01
		net2 = np.prod(X*w2 + b2)
		f2 = functions.sigmoid(net2)
		f2 = f2 * bit_02
		
		#Calculate Net3
		net3 = w3[0]*f1 + w3[1]*f2 + b3
		return net3
		
	
	def setFinalData(self, X_pre):
		if (X_pre.shape[0] >= self.z + 1) and (X_pre.shape[1] == self.m):
			self._final = X_pre[-(self.z + 1):, :]
		else: 
			print('Dimension mismatch - setFinalData function unable to execute.')
		return
	
	def fit_MPSO(self, X, y, d = 30, c1 = 2.0, c2 = 2.0, w = 1.0, maxt = 1000):

		self.MSE_gBest = []

		# Final part of training set to make full prediction of test set
		self._final = X[-(self.z + 1):, :]

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
					predict_res[j] = self.forward_with_decision(p[3*self.m+4 : 3*self.m+4 + 3*self.z+6], x)
# Super particle
# 3rd Layer: Combination
				X_comb = np.hstack((predict[self.z:].reshape(-1,1), predict_res.reshape(-1,1)))
				X_comb = np.hstack((X_comb, np.ones(len(X_comb)).reshape(-1,1)))

				output = np.zeros(len(X_error_train))
				for j, x in enumerate(X_comb):
					output[j] = functions.sigmoid(np.dot(p[-3:], x))
				fitness[i] = mean_squared_error(y[self.z:], output)
				# Check fitness
				if fitness[i] < best_fitness[i]:
					pBest[i] = p[:]
					best_fitness[i] = fitness[i]
# 			if t%100==0:
# 				print('Fitness da iteração:', t  ,' é:',  best_fitness[i] )

			#Choosing the best particle
			if t%10 == 0:
				self.MSE_gBest.append(fitness.min())
			gBest = particles[np.argmin(fitness)]

			# MPSO Parameters
			bad_index = np.argmax(fitness)
			# c1 = ((c1f - c1i)*t/maxt) +c1i
			# c2 = ((c2f - c2i)*t/maxt) +c2i
			# w = w1 + (w2 - w1)*((maxt - t)/maxt)

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
		# Adapt final part of training set to test se
		X_test = np.vstack((self._final, X_test))
		y_test = X_test[1:,-1]

		series = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			series[i] = self.forward_series(self.weight[:3*self.m+4], x)
		
		# Generating residual windows
		resid = functions._error(y_test, series[:-1])
		resid_window = functions.gerar_janelas(tam_janela = self.z-1, serie = resid)

		pred_resid = np.zeros(len(resid_window))
		for i, x in enumerate(resid_window):
			pred_resid[i] = self.forward_series(self.weight[3*self.m+4 : 3*self.m+4 + 3*self.z+4], x)
		X_comb = np.hstack((series[self.z + 1:].reshape(-1,1), pred_resid.reshape(-1,1)))
		X_comb = np.hstack((X_comb, np.ones(len(X_comb)).reshape(-1,1)))

		# Funniest part!
		predict = np.zeros(len(X_comb))
		for i, x in enumerate(X_comb):
			predict[i] = functions.sigmoid(np.dot(self.weight[-3:], x))
		return predict
