import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random
import functions
import matplotlib.pyplot as plt

class Hybrid_ANN:
	def __init__(self, m = 5):
		self.k = 3*m + 4
		self.weight = np.random.rand(self.k)
		return

	def forward(self, weight, X):
		m = int((len(weight)-4)/3)
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
		w1 = 1.0, w2 = 2.0, maxt = 500):
		#MPSO Algorythm
		particles = np.random.rand(d, self.k)
		velocity = np.random.uniform(low = -1.0, high = 1.0, size = (d, self.k))
		pBest = particles[:]
		gBest = self.weight[:]
		best_fitness = np.full(d, np.inf)
		# print(particles)
		for t in range(maxt):
			fitness = np.zeros(d)
			for i, p in enumerate(particles):
				output = np.zeros(len(y))
				for j, x in enumerate(X):
					output[j] = self.forward(p, x)
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
					velocity[i] = np.random.uniform(low = -1.0, high = 1.0, size = self.k)
					particles[i] = np.random.rand(self.k)
					pBest[i] = particles[i]
				else:
					velocity[i] = c1*random.uniform(0, 1)*(pBest[i] - p) + c2*random.uniform(0, 1)*(gBest - p)
					particles[i] = p +velocity[i]
		#Optimal solution. The L&NL-ANN weights will be gBest
		self.weight = gBest[:]
		return

	def predict(self, X_test):
		predict = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			predict[i] = self.forward(self.weight, x)
		return predict

df = pd.read_csv('../arma_model/sales_house.csv')
data = df.iloc[:, -1].values

serie_normal = functions.normalizar_serie(data)
serie_lags = functions.gerar_janelas(tam_janela = 5, serie = serie_normal)
X_train, y_train, X_test, y_test = functions.split_serie_with_lags(serie = serie_lags, perc_train = 0.8, perc_val = 0)

model = Hybrid_ANN(m = 5)

# c = np.zeros(2)
# best_mse = np.inf
# best_weight = []
# for c1 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
# 	for c2 in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
# 		model.fit_MPSO(X = X_train, y = y_train, d= 30,
# 			c1 = c1, c2 = c2 ,w1 = 1.0, w2 = 2.0, maxt = 2000)
# 		y_pred = model.predict(X_test)
# 		temp_mse = mean_squared_error(y_test, y_pred)
# 		if temp_mse < best_mse:
# 			best_mse = temp_mse
# 			best_weight = model.weight[:]
# 			c[0] = c1
# 			c[1] = c2

# print('Best MSE:',best_mse,'c1:',c[0],'c2:',c[1])

model.fit_MPSO(X = X_train, y = y_train, d= 30,
	c1i = 1.0, c1f = 2.0, c2i = 2.5, c2f = 3.5, maxt = 2000)
y_pred = model.predict(X_test)
# print('Prediction:', y_pred)
# print('Test:', y_test)

plt.plot(y_test, label = 'Original')
plt.plot(y_pred, label = 'Prediction', color = 'red')
plt.legend()
plt.show()