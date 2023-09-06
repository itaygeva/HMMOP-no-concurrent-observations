import numpy as np
from hmmlearn import hmm
import itertools

import itertools

original_array = [2, 2, 3]  # Replace this with your array
n = len(original_array)

# Generate all permutations of the original array
permutations = list(itertools.permutations(original_array, n))

# Concatenate the permutations to create the new array
new_array = [item for sublist in permutations for item in sublist]

print(new_array)

original_array = [1, 2, 3]  # Replace this with your array
n = 3  # Replace this with the number of times you want to concatenate

concatenated_array = original_array * (n**
                                       2)
print(concatenated_array)
