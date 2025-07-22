def find_max(lst):
    try:
        max_val = lst[0]
        for num in lst[1:]:
            if num > max_val:
                max_val = num
        return max_val
    except IndexError:
        return None