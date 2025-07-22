def is_palindrome(s):
    rev = ''
    for char in s:
        rev = char + rev
    return s == rev