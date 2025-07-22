def fibonacci(n):
    def fib(x):
        if x <= 1:
            return x
        return fib(x-1) + fib(x-2)
    return list(map(fib, range(n)))