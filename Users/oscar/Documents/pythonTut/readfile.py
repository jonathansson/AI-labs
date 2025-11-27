def parse(file):

    input = open(file, "r")
    input_lines = input.readlines()

    transition_split = str.split(' ', input_lines[0])
    emission_split = str.split(' ', input_lines[1])
    initial_split = str.split(' ', input_lines[2])

    print(transition_split)
    