def reverse_string(s):
    return ''.join(s[-i] for i in range(1, len(s)+1))