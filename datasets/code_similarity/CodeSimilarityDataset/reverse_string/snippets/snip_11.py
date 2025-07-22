def reverse_string(s):
    lst = [s[i] for i in range(len(s))]
    lst.reverse()
    return ''.join(lst)