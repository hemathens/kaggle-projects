def find_max(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return lst[0] if lst[0] > find_max(lst[1:]) else find_max(lst[1:])