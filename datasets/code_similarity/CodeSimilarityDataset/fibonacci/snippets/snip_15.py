def fibonacci(n):
    if n == 0:
        return []
    seq = [0] * n
    seq[0] = 0
    if n > 1:
        seq[1] = 1
        for i in range(2, n):
            seq[i] = seq[i-1] + seq[i-2]
    return seq