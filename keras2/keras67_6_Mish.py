import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))
    # x * tanh(softplus(x))
    # softplus(x) = log( 1 + exp(x))


y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()
