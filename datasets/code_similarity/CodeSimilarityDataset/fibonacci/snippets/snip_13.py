def fibonacci(n, a=0, b=1, seq=None):
    if seq is None:
        seq = []
    if n == 0:
        return seq
    seq.append(a)
    return fibonacci(n-1, a=b, b=a+b, seq=seq)

print(fibonacci(10))  # Example usage