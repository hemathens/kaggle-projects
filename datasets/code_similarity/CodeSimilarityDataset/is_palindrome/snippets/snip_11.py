def is_palindrome(s):
    return all(a == b for a, b in zip(s, reversed(s)))