def find_max(lst):
    max_index = 0
    for i in range(1, len(lst)):
        if lst[i] > lst[max_index]:
            max_index = i
    return lst[max_index]