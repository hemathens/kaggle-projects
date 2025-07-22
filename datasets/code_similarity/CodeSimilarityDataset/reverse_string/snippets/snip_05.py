def reverse_string(s):
    length = len(s)
    chars = list(s)
    for i in range(length // 2):
        chars[i], chars[length - i - 1] = chars[length - i - 1], chars[i]
    return ''.join(chars)