import numpy as np
from error_functions import mean_squared_error
import math
import random
import functions

class LNL_ANN:
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

	def fit_MPSO(self, X, y, d = 30, c1 = 2.0, c2 = 2.0, w = 1, maxt = 500):

		self.MSE_gBest = []
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
			if t%10 == 0:
				self.MSE_gBest.append(fitness.min())
			gBest = particles[np.argmin(fitness)]
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
			
		#Optimal solution. The L&NL-ANN weights will be gBest
		self.weight = gBest[:]
		return

	def predict(self, X_test):
		predict = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			predict[i] = self.forward(self.weight, x)
		return predict




class Hybrid_ANN:
	def __init__(self, pre_weight: np.ndarray):
		m = int((len(pre_weight)-4)/3)
		self.k = len(pre_weight) + 3
		self.weight = pre_weight[:]
		self.weight = np.insert(self.weight, 3*m+1, np.random.uniform())
		self.weight = np.insert(self.weight, 2*m+1, np.random.uniform())
		self.weight = np.insert(self.weight, m, np.random.uniform())
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
		w1 = 0.1, w2 = 1.0, maxt = 500):
		#MPSO Algorythm
		change = 3
		m = int((len(self.weight)-4)/3)
		particles = np.random.rand(d, change) #Experimental
		velocity = np.random.uniform(low = -1.0, high = 1.0, size = (d, change)) #Experimental
		pBest = particles[:]
		gBest = particles[0]
		best_fitness = np.full(d, np.inf)
		aux_weight = self.weight[:]
		# print(particles)
		for t in range(maxt):
			fitness = np.zeros(d)
			for i, p in enumerate(particles):
				aux_weight[[m, 2*m+2, 3*m+3]] = p[:] #Ligado ao 'change'
				output = np.zeros(len(y))
				for j, x in enumerate(X):
					output[j] = self.forward(aux_weight, x)
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
					velocity[i] = np.random.uniform(low = -1.0, high = 1.0, size = change)
					particles[i] = np.random.rand(change)
					pBest[i] = particles[i]
				else:
					velocity[i] = w*velocity[i] + c1*random.uniform(0, 1)*(pBest[i] - p) + c2*random.uniform(0, 1)*(gBest - p)
					particles[i] = p +velocity[i]
		#Optimal solution. The L&NL-ANN weights will be gBest
		self.weight[[m, 2*m+2, 3*m+3]] = gBest[:] #Ligado ao 'change'
		return

	def predict(self, X_test):
		predict = np.zeros(len(X_test))
		for i, x in enumerate(X_test):
			predict[i] = self.forward(self.weight, x)
		return predict
