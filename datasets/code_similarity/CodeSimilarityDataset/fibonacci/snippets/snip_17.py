def fibonacci(n):
    seq = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)
        next_val = a + b
        a, b = b, next_val
    return seq