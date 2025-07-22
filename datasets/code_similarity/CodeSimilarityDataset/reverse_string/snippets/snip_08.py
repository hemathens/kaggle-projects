def reverse_string(s):
    reversed_str = ''
    index = len(s)
    while index > 0:
        reversed_str += s[index-1]
        index -= 1
    return reversed_str