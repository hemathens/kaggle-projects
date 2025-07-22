def find_max(lst):
    for i in range(len(lst)-1):
        if lst[i+1] < lst[i]:
            lst[i], lst[i+1] = lst[i+1], lst[i]
    return lst[-1]