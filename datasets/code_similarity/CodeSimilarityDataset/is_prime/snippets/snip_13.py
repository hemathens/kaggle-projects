def is_prime(n):
    if n in [2, 3, 5, 7, 11, 13, 17, 19]:
        return True
    if n < 2 or n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True