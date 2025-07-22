def find_max(lst):
    i = 0
    max_val = lst[0]
    while i < len(lst):
        if lst[i] > max_val:
            max_val = lst[i]
        i += 1
    return max_val