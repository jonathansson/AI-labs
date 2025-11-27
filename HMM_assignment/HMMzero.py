import sys

file = sys.stdin.read()


def parse(file):

    file_split = file.split('\n')

    transition_string = file_split[0]
    emission_string = file_split[1]
    initial_string = file_split[2]

    transition = list(map(float, transition_string.split()))
    emission = list(map(float, emission_string.split()))
    initial = list(map(float, initial_string.split()))

    return transition, emission, initial

# matrix a * marix b
def matrix_mult(a, b):

    a_rows = int(a[0])
    a_cols = int(a[1])
    b_rows = int(b[0])
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



transition, emission, initial = parse(file)
init_tran = matrix_mult(initial, transition)
print(init_tran)
init_tran_emis = matrix_mult(init_tran, emission)
print(init_tran_emis)
