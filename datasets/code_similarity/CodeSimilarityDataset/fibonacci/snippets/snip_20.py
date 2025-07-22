import numpy as np

def fibonacci(n):
    if n <= 0:
        return []
    F = np.array([0, 1], dtype=int)
    for _ in range(2, n):
        F = np.append(F, F[-1] + F[-2])
    return F.tolist()