def is_palindrome(s):
    mid = len(s) // 2
    first_half = s[:mid]
    second_half = s[-mid:]
    return first_half == second_half[::-1]