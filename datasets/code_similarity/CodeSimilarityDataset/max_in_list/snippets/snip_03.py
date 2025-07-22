from functools import reduce

def find_max(lst):
    return reduce(lambda a, b: a if a > b else b, lst)