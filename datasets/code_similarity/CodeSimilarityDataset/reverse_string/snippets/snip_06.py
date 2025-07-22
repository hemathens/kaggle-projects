def reverse_string(s):
    return ''.join(s[i] for i in range(len(s)-1, -1, -1))