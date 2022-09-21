from utils import sigmoid

import numpy as np


test = np.ones(shape=(6,1))
for i in range(6):
	test[i] = i 



test1 = np.ones(shape=(5,5))
test1[4][2] = 2
test1[2][3] = 0.02
targets = np.ones(shape=[5,1])


y = sigmoid(np.dot(test1, test[:len(test) - 1]) + test[len(test) - 1])

print(y, "hi im goomba")

for i in range(5):
	targets[i] = i + 1 
print(np.dot(-targets.T,np.log(y)))


a = np.array([[5, 1, 3], [1, 1, 1], [1, 2, 1]])
b = np.array([1, 2, 3])
x = np.random.randn(4 + 1, 1)
x[4] = 34
print(x[4])
