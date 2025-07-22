def find_max(lst):
    return lst[[i for i in range(len(lst)) if all(lst[i] >= x for x in lst)][0]]