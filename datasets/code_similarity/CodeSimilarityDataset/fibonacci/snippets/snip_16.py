from itertools import islice

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def fib_sequence(n):
    return list(islice(fibonacci(), n))