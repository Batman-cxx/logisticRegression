# -*- coding: utf-8 -*-
import numpy as np 
import math
import matplotlib.pyplot as plt 
import matplotlib.pylab as pl

# watermelon dataset
data = np.array([[1,0.697,0.460,1],
	[2,0.774,0.376,1],
	[3,0.634,0.264,1],
	[4,0.608,0.318,1],
	[5,0.556,0.215,1],
	[6,0.403,0.237,1],
	[7,0.481,0.149,1],
	[8,0.437,0.211,1],
	[9,0.666,0.091,0],
	[10,0.243,0.267,0],
	[11,0.245,0.057,0],
	[12,0.343,0.099,0],
	[13,0.639,0.161,0],
	[14,0.657,0.198,0],
	[15,0.360,0.370,0],
	[16,0.593,0.042,0],
	[17,0.719,0.103,0]])

# print data
y = data[:,3]
train_x = data[:,1:3]
train_x = np.c_[train_x,np.ones(len(y))]    # [x;1]
train_theta = np.array([1.0,1.0,1.0])	# [w;b]
iteration = 10000
num_samples = len(y)


def sigmoid(theta, x): 
	return 1.0/(1 + math.exp(-np.dot(theta,x)))

def converge(new_theta, theta):    # 是否收敛
	sub_theta = new_theta - theta
	for i,j in zip(sub_theta, theta):
		if(i!= 0 and abs(i/j)>0.001):
			return False
	return True

def gradientDescend(theta,x):	# 梯度下降法
	g = 0
	for n in range(num_samples):
		g += -(y[n] - sigmoid(theta,x[n])) * train_x[n]
	alpha = 0.05
	delta = g * alpha
	return delta

def newton(theta,x):	# 牛顿法
	alpha = np.mat(np.zeros((3,3)))
	g = 0
	for n in range(num_samples):
		g += -(y[n] - sigmoid(theta,x[n])) * train_x[n]
		alpha += np.mat(train_x[n]).T * np.mat(train_x[n]) * sigmoid(theta,x[n]) * (1 - sigmoid(theta,x[n]))
	result = np.mat(g) * alpha.I
	delta = result.getA()
	delta = delta.reshape(-1)
	return delta

def logistic_regression(theta,x):		
	for i in range(iteration):
		print("iteration {0}".format(i))
		# delta = gradientDescend(theta,x)
		delta = newton(theta,x)
		new_theta = theta - delta
		if(converge(new_theta,theta)):
			print("converge")
			break
		theta = new_theta
	print theta
	return theta

def predict(X,theta):
	m,n = np.shape(X)
	y = np.zeros(m)
	for i in range(m):
		result = sigmoid(theta,X[i])   # X [x,1] 1*(d+1)
		if result > 0.5:
			y[i] = 1
		else:
			y[i] = 0
	return y

train_theta = logistic_regression(train_theta, train_x)

pre = predict(train_x, train_theta)
print pre

density_min, density_max = train_x[:,0].min()-0.1, train_x[:,0].max()+0.1
sugar_min, sugar_max = train_x[:,1].min()-0.1, train_x[:,1].max()+0.1
dd,ss = np.meshgrid(np.arange(density_min, density_max, 0.001),np.arange(sugar_min, sugar_max, 0.001))

w = np.c_[dd.ravel(), ss.ravel()]
w = np.c_[w,np.ones(len(w)).ravel()]
z = predict(w,train_theta)
# print np.shape(z)

z = z.reshape(dd.shape)

plt.contourf(dd, ss, z, cmap=pl.cm.Paired)
plt.title("watermelon")
plt.xlabel("density")
plt.ylabel("sugar_ration")
plt.scatter(data[y==0,1],data[y==0,2],color = 'k',marker = 'o',s = 100, label = "bad")
plt.scatter(data[y==1,1],data[y==1,2],color = 'g',marker = 'o',s = 100, label = "good")
plt.legend(loc = 'upper right')
plt.show()

