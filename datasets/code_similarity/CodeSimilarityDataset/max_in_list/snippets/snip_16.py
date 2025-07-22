def find_max(lst):
    def compare(a, b):
        return a if a > b else b
    max_val = lst[0]
    for num in lst[1:]:
        max_val = compare(max_val, num)
    return max_val