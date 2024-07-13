def swap_list(new_list):
    size = len(new_list)

    # swapping the first and last number

    temp = new_list[0]
    new_list[0] = new_list[size - 1]
    new_list[size - 1] = temp

    return new_list

new_list = [12,34,56,67,89]

swap_list(new_list)



