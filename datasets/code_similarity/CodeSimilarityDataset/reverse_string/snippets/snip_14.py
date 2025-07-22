def reverse_string(s):
    return ''.join(s[len(s)-i-1] for i in range(len(s)))