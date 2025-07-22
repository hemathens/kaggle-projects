def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fib_sequence(n):
    return [fibonacci(i) for i in range(n)]