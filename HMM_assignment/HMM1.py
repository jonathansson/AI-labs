import sys

file = sys.stdin.read()

def parse(file):

    file_split = file.split('\n')

    transition_string = file_split[0]
    emission_string = file_split[1]
    initial_string = file_split[2]
    sequence_string = file_split[3]

    transition = list(map(float, transition_string.split()))
    emission = list(map(float, emission_string.split()))
    initial = list(map(float, initial_string.split()))
    sequence = list(map(float, sequence_string.split()))

    return transition, emission, initial, sequence

# matrix a * marix b
def matrix_mult(a, b):

    a_rows = int(a[0])
    a_cols = int(a[1])
    b_cols = int(b[1])
    res=[a_rows, b_cols]

    # for each row in a
    for i in range(a_rows):
        # for each column in b
        for j in range(b_cols):
            sum = 0
            # dot product of row i of a with column j of b
            for k in range(a_cols):
                element_a = a[2 + i * a_cols + k]  # element at row i, col k of matrix a
                element_b = b[2 + k * b_cols + j]  # element at row k, col j of matrix b
                sum += element_a * element_b
            res.append(sum)
    
    return res

# Element-wise multiplication
def matrix_element_mult(a, b):
    res = [a[0], a[1]]
    for x in range(2, len(a)):
        res.append(a[x] * b[x])
    return res

# Gets a column from a matrix
def getColumn(matrix, column_nr, nr_of_columns):
    rows = int(matrix[0])
    col = []
    col.append(1)
    col.append(rows)
    for x in range(2 + int(column_nr), len(matrix), int(nr_of_columns)):
        col.append(matrix[x])
    return col

def forward_algo(transition, emission, initial, sequence):
        
    o1_index = int(sequence[1])
    nr_states = int(emission[0])
    nr_emissions = int(emission[1])
    
    # Get b(o1)
    b_o1 = getColumn(emission, o1_index, nr_emissions)
    
    # Alpha_1 = pi * b(o1), element-wise multiplication
    alpha = matrix_element_mult(initial, b_o1)
        
    for x in range(2, len(sequence)):
        obs_index = int(sequence[x])
            
        # Transition: alpha x A
        alpha_trans = matrix_mult(alpha, transition)
        # Emission: Get column for the current observation
        b_obs = getColumn(emission, obs_index, nr_emissions)
        # Update alpha: alpha_trans * b_obs
        alpha = matrix_element_mult(alpha_trans, b_obs)
        
    # Sum the data-part of alpha (index 2 and forward)
    return sum(alpha[2:])
            
transition, emission, initial, sequence = parse(file)
print(forward_algo(transition, emission, initial, sequence))


