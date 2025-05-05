
import numpy as np

def find_numbers_in_lists(list_of_lists, numbers_to_find):
    """
    Find numbers in a list of lists and return a list of lists showing indices of lists containing those numbers.

    Parameters:
    - list_of_lists (list): A list of lists to search for numbers.
    - numbers_to_find (list): A list of numbers to find within the lists.

    Returns:
    - indices_containing_numbers (list of lists): A list of sublists. Each sublist contains the indices of lists in list_of_lists
    that contain a specific number from numbers_to_find. Empty sublists are created for numbers not found in any list.

    Example:
    >>> list_of_lists = [[1, 2, 3], [4, 5, 6], [6, 7, 8, 9], [10, 11, 12]]
    >>> numbers_to_find = [6, 10, 12, 15]
    >>> result = find_numbers_in_lists(list_of_lists, numbers_to_find)
    >>> print(result)
    Output: [[1, 2], [3], [], []]
    """
    
    indices_containing_numbers = []
    
    for num in numbers_to_find:
        indices = [index for index, lst in enumerate(list_of_lists) if num in lst]
        indices_containing_numbers.append(indices)

    return indices_containing_numbers


def select(lst, indices,concatenate=False):
    list_out = []
    for i in indices:
        list_out.append(lst[i])
    if concatenate:
        list_out = np.concatenate(list_out)
    return list_out


def find_single_number_in_list(input_list, target_number):
    """
    Find the index of a target number in a list of numbers.

    Parameters:
    - input_list (list): A list of numbers to search.
    - target_number (int or float): The number to find in the list.

    Returns:
    - indexes (list): A list of indices where the target number is found.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 2, 7, 8, 9]
    >>> target_number = 2
    >>> result = find_single_number_in_list(input_list, target_number)
    >>> print(result)
    Output: [1, 5]
    """
    # Use a list comprehension to find the indexes of matching numbers
    indexes = [i for i, number in enumerate(input_list) if number == target_number]
    return indexes

def find_numbers_in_list(input_list, target_numbers):
    """
    Find the indices of target numbers in a list of numbers.

    Parameters:
    - input_list (list): A list of numbers to search.
    - target_numbers (list): A list of numbers to find in the input list.

    Returns:
    - indexes (list of lists): A list of lists containing the indices of target numbers in the input list.

    Example:
    >>> input_list = [1, 2, 3, 4, 5, 2, 7, 8, 9]
    >>> target_numbers = [2, 7]
    >>> result = find_numbers_in_list(input_list, target_numbers)
    >>> print(result)
    Output: [[1, 5], [6]]
    """
    # Initialize an empty list to store indexes
    indexes = []

    # Iterate over each target number
    for target_number in target_numbers:
        # Find the indexes of the target number in the input list
        target_indexes = [i for i, number in enumerate(input_list) if number == target_number]
        # Append the list of indexes to the main list
        indexes.append(target_indexes)

    return indexes


def find_single_string_in_list(input_list, target_string):
    # Use a list comprehension to find the indexes of matching strings
    indexes = [i for i, string in enumerate(input_list) if string == target_string]
    return indexes


def find_list_strings_in_list(main_list, strings_to_find):
    """
    Find strings in a list and return a list of indices showing the positions of those strings.

    Parameters:
    - main_list (list): A list of strings to search within.
    - strings_to_find (list): A list of strings to find within the main_list.

    Returns:
    - indices_containing_strings (list of lists): A list of lists of indices. Each sublist contains the indices
      of a string in strings_to_find found in main_list. If a string is not found, the corresponding sublist
      will be empty.

    Example:
    >>> main_list = ['apple', 'banana', 'orange', 'grape', 'kiwi', 'pear', 'melon', 'strawberry', 'blueberry']
    >>> strings_to_find = ['orange', 'kiwi', 'watermelon', 'grapefruit']
    >>> result = find_strings_in_list(main_list, strings_to_find)
    >>> print(result)
    Output: [[2], [4]]
    """
    
    indices_containing_strings = [[] for _ in range(len(strings_to_find))]

    for idx, item in enumerate(main_list):
        if item in strings_to_find:
            index = strings_to_find.index(item)
            indices_containing_strings[index].append(idx)
            
    return indices_containing_strings


'''

def find_strings_in_list(main_list, strings_to_find):
    """
    Find strings in a list and return a list of indices showing the positions of those strings.

    Parameters:
    - main_list (list): A list of strings to search within.
    - strings_to_find (list): A list of strings to find within the main_list.

    Returns:
    - indices_containing_strings (list): A list of indices. Each element contains the index of a string in main_list.
      If a string is not found, the corresponding index is np.nan.

    Example:
    >>> main_list = ['apple', 'banana', 'orange', 'grape', 'kiwi', 'pear', 'melon', 'strawberry', 'blueberry']
    >>> strings_to_find = ['orange', 'kiwi', 'watermelon', 'grapefruit']
    >>> result = find_strings_in_list(main_list, strings_to_find)
    >>> print(result)
    Output: [2, 4, np.nan, np.nan]
    """
    
    indices_containing_strings = []

    for string_to_find in strings_to_find:
        try:
            index = main_list.index(string_to_find)
        except ValueError:
            index = np.nan
        indices_containing_strings.append(index)

    return np.array(indices_containing_strings)

import numpy as np

def find_strings_in_list(main_list, strings_to_find):
    """
    Find strings in a list and return a list of indices showing the positions of those strings.

    Parameters:
    - main_list (list): A list of strings to search within.
    - strings_to_find (list): A list of strings to find within the main_list.

    Returns:
    - indices_containing_strings (list): A list of indices. Each element contains the index of a string in main_list.
      If a string is not found, the corresponding index is np.nan.

    Example:
    >>> main_list = ['apple', 'banana', 'orange', 'grape', 'kiwi', 'pear', 'melon', 'strawberry', 'blueberry']
    >>> strings_to_find = ['orange', 'kiwi', 'watermelon', 'grapefruit']
    >>> result = find_strings_in_list(main_list, strings_to_find)
    >>> print(result)
    Output: [2, 4, np.nan, np.nan]
    """
    
    indices_containing_strings = []

    for idx, item in enumerate(main_list):
        if item in strings_to_find:
            indices_containing_strings.append(idx)
            
    return np.array(indices_containing_strings)


'''