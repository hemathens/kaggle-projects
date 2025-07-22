def find_max(lst):
    for num in lst:
        is_max = True
        for other in lst:
            if other > num:
                is_max = False
                break
        if is_max:
            return num