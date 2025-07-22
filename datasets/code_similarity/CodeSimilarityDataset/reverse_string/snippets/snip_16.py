def reverse_string(s):
    return ''.join('' if not c else c for c in reversed(s))