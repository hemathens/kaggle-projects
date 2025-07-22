from functools import reduce

def fibonacci(n):
    if n <= 0:
        return []
    return reduce(lambda x, _: x + [x[-1] + x[-2]], range(n-2), [0, 1])[:n]