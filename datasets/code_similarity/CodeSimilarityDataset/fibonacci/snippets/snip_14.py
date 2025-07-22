import math

def fibonacci(n):
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    return round(phi**n / sqrt5)

def fib_sequence(n):
    return [fibonacci(i) for i in range(n)]