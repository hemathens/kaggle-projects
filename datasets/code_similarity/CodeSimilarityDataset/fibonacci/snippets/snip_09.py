import numpy as np

def fibonacci(n):
    def power(mat, n):
        result = np.identity(2, dtype=int)
        while n > 0:
            if n % 2 == 1:
                result = np.dot(result, mat)
            mat = np.dot(mat, mat)
            n //= 2
        return result

    F = np.array([[1, 1], [1, 0]], dtype=int)
    result = [0, 1]
    for i in range(n):
        if i == 0:
            result.append(0)
        elif i == 1:
            result.append(1)
        else:
            result.append((power(F, i-1)[0][0]))
    return result[:n]