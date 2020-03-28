def nearest_square_side(n):
    i = 1
    while i ** 2 < n:
        i += 1
    return i
