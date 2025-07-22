import heapq

def find_max(lst):
    return heapq.nlargest(1, lst)[0]