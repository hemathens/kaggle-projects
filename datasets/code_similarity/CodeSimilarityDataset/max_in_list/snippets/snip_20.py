def find_max(lst):
    freq = {}
    for num in lst:
        freq[num] = freq.get(num, 0) + 1
    return max(freq.keys())