import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5)

def selu(x, lambda_=1.0507, alpha=1.67326):
    return np.where(x >= 0, lambda_ * x, lambda_ * (alpha * np.exp(x) - alpha))


y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()
