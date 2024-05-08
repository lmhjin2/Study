# SiLu (Sigmoid-weighted Linear Unit) = Swish

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5)
def silu(x):
    return x * (1 / (1 + np.exp(-x)))
# == x * (1 / (1 + e^-x) == x * sigmoid(x)
# silu = lambda x : x * (1 / (1 + np.exp(-x)))
y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()