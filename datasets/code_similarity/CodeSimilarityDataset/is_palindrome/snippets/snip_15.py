def is_palindrome(s):
    stack = list(s)
    return s == ''.join(stack.pop() for _ in range(len(s)))