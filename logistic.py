import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ERRORS = []

def hypothesis(coefficients, sample):
	acum = np.dot(np.array(coefficients),np.array(sample))  
	return acum


def sigmoid(x):
	return 1/(1+ math.exp(-1 * x))


def cross_entropy(coefficients, samples, y):
	error_acum =0
	error = 0
	for i in range(len(samples)):
		hyp = sigmoid(hypothesis(coefficients,samples[i]))

		if(y[i] == 1): 
			if(hyp ==0):
				hyp = .0001
			error = (-1)*math.log(hyp)

		if(y[i] == 0):
			if(hyp ==1):
				hyp = .9999
			error = (-1)*math.log(1-hyp)

		error_acum = error_acum +error 
	mean_error_param=error_acum/len(samples)
	ERRORS.append(mean_error_param)

	return mean_error_param


def update_params(coefficients, samples, y, alpha):
	temp = list(coefficients)
	for j in range(len(coefficients)):
		acum = 0 
		for i in range(len(samples)):
			error = sigmoid(hypothesis(coefficients,samples[i]) - y[i])
			acum = acum + error*samples[i][j] 
		temp[j] = coefficients[j] - alpha*(1/len(samples))*acum  
	return temp




columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age", "Outcome"]
df = pd.read_csv('diabetes2.csv',names = columns)
print(len(columns))
coefficients = [0,0,0,0,0,0,0,0,0]
samples = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI","DiabetesPedigreeFunction","Age"]]
samples.insert(0,'Beta',1)

y = df["Outcome"].values.tolist()
y = np.array(y)
y = y[1:]
y = y.astype(np.double)
y = y.tolist()


samples = samples.values.tolist()
samples = np.array(samples)
samples = samples[1:,:]
samples = samples.astype(np.double)
samples = samples.tolist()

print(samples)

alpha =.01
print ("original samples:")
print (samples)
# Scaling
samples = np.array(samples)
samples = np.divide(samples, 1000)
samples = samples.tolist()
print ("scaled samples:")
print (samples)

number_epochs = 10000
for i in range(number_epochs):
	old_coefficients = list(coefficients)
	coefficients=update_params(coefficients, samples,y,alpha)	
	error = cross_entropy(coefficients, samples, y)



print (coefficients)
plt.plot(ERRORS)
plt.show()

print("Parameters ",coefficients)

query = 1
while query != 0:
	print("Enter the following in order separated by spaces or 0 to quit")
	print("Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BME, DiabetesPedigreeFunction, Age")
	query = list(map(float,input().split()))
	# For the beta
	query.insert(0,1)

	# Scaling
	query = np.array(query)
	query = np.divide(query, 1000)
	query = query.tolist()

	# Result of query
	result = sigmoid(hypothesis(coefficients,query))
	print("-------------------")
	print("Result: ",result)

	if result < 0.5:
		print("Low probability of having diabetes")
	else:
		print("High probability of having diabetes")
	print("-------------------")

# 6 148 72 35 0 33.6 0.627 50 -> 1
# 3 78 50 32 88 31 0.248 26 -> 1

# 11 143 94 33 146 36.6 0.254 51 -> 1

# 10 139 80 0 0 27.1 1.441 57 -> 0

# 3 158 76 36 245 31.6 0.851 28 -> 1

# 11 155 76 28 150 33.3 1.353 51 -> 1

# 12 140 82 43 325 39.2 0.528 58 -> 1