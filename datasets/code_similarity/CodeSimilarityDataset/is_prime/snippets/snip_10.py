def is_prime(n):
    if n < 2:
        return False
    return list(filter(lambda x: n % x == 0, range(2, int(n**0.5)+1))) == []