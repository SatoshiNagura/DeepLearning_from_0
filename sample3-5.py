import numpy as np

def SoftMaxFunction(Array):
    return np.exp(Array) / np.sum(np.exp(Array))

a = np.array([0.3, 2.9, 4.0])
print(SoftMaxFunction(a))
