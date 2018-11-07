import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import random

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
		#Activation Functions
		f1 = net1
		f2 = 1/(1 + math.exp(-net2))
		#Calculate Net3
		net3 = w3[0]*f1 + w3[1]*f2 + b3
		return net3

	def fit_MPSO(self, X, y, d = 10, c1i = 2, c1f = 3, 
		c2i = 2, c2f = 3, w1 = 1, w2 = 2, maxt = 1000):
		#MPSO Algorythm
		particles = np.random.rand(d, self.k)
		velocity = np.random.rand(d, self.k)
		pBest = particles[:]
		gBest = self.weight[:]
		old_fitness = np.full(len(y), np.inf)
		print(particles)
		for t in range(1, maxt+1):
			fitness = np.zeros(len(y))
			for i, p in enumerate(particles):
				output = np.zeros(len(y))
				for j, x in enumerate(X):
					output[j] = self.forward(p, x)
				fitness[i] = mean_squared_error(y, output)
				if fitness[i] < old_fitness[i]:
					pBest[i] = p
			old_fitness = fitness[:]
			gBest = particles[np.argmin(fitness)]
			c1 = ((c1f - c1i)*t/maxt) +c1i
			c2 = ((c2f - c2i)*t/maxt) +c2i
			w = ((w2 - w1)*(maxt - t)/maxt) + w1
			for i, p in enumerate(particles):
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