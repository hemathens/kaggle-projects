def is_prime(n):
    return n > 1 and not ('a' * n).replace('a', '', 1).find('a' * 2) != -1