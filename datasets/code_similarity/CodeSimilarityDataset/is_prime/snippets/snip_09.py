def is_prime(n):
    if n <= 1:
        return False
    divisors = [i for i in range(1, n+1) if n % i == 0]
    return len(divisors) == 2   