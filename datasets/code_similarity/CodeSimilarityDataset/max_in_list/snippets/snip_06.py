def find_max(lst):
    max_val = float('-inf')
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val