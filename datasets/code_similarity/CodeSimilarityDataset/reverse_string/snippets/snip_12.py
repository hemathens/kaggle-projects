def reverse_string(s):
    rev = ''
    for i in range(len(s)-1, -1, -1):
        rev += s[i]
    return rev