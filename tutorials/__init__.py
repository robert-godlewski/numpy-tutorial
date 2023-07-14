import numpy as np


# Main app runner
def run() -> None:
    # Creating a basic array
    a = np.array([1,2,3,4,5,6])
    print(a)

    # Creating an array with 0s
    z = np.zeros(2)
    print(z)

    # Creating an array filled with 1s
    ones = np.ones(2)
    print(ones)

    # Creating an empty array with 2 elements
    empty = np.empty(2)
    print(empty)

    # Creating an array with a range
    range_arr = np.arange(4)
    print(range_arr)

    # Creating an array with a special range
    # first number = 2
    # last number = 9
    # iritive increment = 2
    spe_range = np.arange(2,9,2)
    print(spe_range)
