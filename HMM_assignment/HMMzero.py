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
    x = 0

    a_rows = a[0]
    a_cols = a[1]
    b_rows = b[0]
    b_cols = b[1]
    res=[a_rows, b_cols]

    while(x < int(a_cols)):
        sum = 0
        y = 0
        while(y < (b_rows)):
            print(y)
            sum += a[y + 2] * b[int(x + (y * b_cols))]
            y += 1
        res.append(sum)
        x += 1
    
    return res



transition, emission, initial = parse(file)
gubbe = matrix_mult(initial, transition)
print(gubbe)
tant = matrix_mult(gubbe, emission)
print(tant)
