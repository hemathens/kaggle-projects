def fibonacci(n):
    return [int(((1.618034)**i - (1 - 1.618034)**i)/5**0.5) for i in range(n)]