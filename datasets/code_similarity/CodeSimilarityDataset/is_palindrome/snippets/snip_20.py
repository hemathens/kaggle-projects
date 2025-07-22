def is_palindrome(s):
    return (lambda x: x == x[::-1])(s)