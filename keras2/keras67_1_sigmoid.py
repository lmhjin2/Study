import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1/ (1 + np.exp(-x))

sigmoid = lambda x : 1 / (1+np.exp(-x))

x = np.range([-5, 5, 0,1])
print(x, len(x))
# print(sigmoid(x))

y = simgoind(x)
plt.plot(x,y)
plt.grid()
plt.show()
