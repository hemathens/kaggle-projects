def is_palindrome(s):
    try:
        return s == s[::-1]
    except Exception:
        return False