def find_max(lst):
    max_val = lst[0]
    [max_val := x if x > max_val else max_val for x in lst]
    return max_val