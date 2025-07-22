def is_prime(n):
    return n > 1 and not any(n % i == 0 for i in range(2, int(n**0.5)+1))