import math


def log10_0(number):
    if number == 0:
        return 0
    return math.log10(number)


def linear_interpolate_zeros(values: list[int]):
    # if value inbetween non zero values is zero, interpolate
    # check every value except first and last
    non_zerod = []
    for i, value in enumerate(values):
        # ignore non zeros
        if value:
            non_zerod.append(value)
            continue

        # find neighbors in both directions, can be done much faster, but works for now
        nearest_left_neighbor = 0
        nearest_right_neighbor = len(values) - 1
        for j, neighbor in enumerate(values):
            if neighbor != 0:
                if nearest_left_neighbor < j < i:
                    nearest_left_neighbor = j
                if i < j:
                    nearest_right_neighbor = j
                    break

        # make sure neighbors are non zero
        start_value = values[nearest_left_neighbor]
        end_value = values[nearest_right_neighbor]
        if start_value and end_value:
            gradient = (end_value - start_value) / (nearest_right_neighbor - nearest_left_neighbor)
            non_zerod.append(start_value + gradient * (i - nearest_left_neighbor))
        else:
            non_zerod.append(value)
    return non_zerod


