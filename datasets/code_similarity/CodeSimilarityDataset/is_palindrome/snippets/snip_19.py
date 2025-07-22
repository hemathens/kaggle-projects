def is_palindrome(s):
    s = s.casefold()
    return s == s[::-1]