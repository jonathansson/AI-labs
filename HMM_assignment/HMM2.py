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


def viterbi_algo(transition, emission, initial, sequence):
    nr_states = int(emission[0])
    nr_emissions = int(emission[1])
    nr_obs_seq = int(sequence[0])

    def get_A(i, j):
        return transition[2 + i * nr_states + j]
    
    def get_B(i, k):
        return emission[2 + i * nr_emissions + k]
    
    def get_pi(i):
        return initial[2 + i]

    deltas = [] 
    back_pointers = []

    # delta_1(i) = pi_i * b_i(o_1)
    first_obs = int(sequence[1])
    delta_1 = []
    for i in range(nr_states):
        delta_1.append(get_pi(i) * get_B(i, first_obs))
    
    deltas.append(delta_1)
    back_pointers.append([0] * nr_states)

    # Iterate t from 2 to T (indices 1 to T-1 in 0-indexed lists)
    # sequence array has structure [Length, obs1, obs2...] so obs at t is sequence[t+1]
    for t in range(1, nr_obs_seq):
        obs = int(sequence[t+1])
        curr_delta = []
        curr_back_pointer = []

        for i in range(nr_states):
            max_prob = -1.0
            best_prev_state = 0
            
            # Calculate max(delta_{t-1}(j) * a_{ji})
            for j in range(nr_states):
                # Prob of being in j prev step * transition j -> i
                prob = deltas[t-1][j] * get_A(j, i)
                
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = j
            
            # Multiply by observation probability b_i(o_t)
            curr_delta.append(max_prob * get_B(i, obs))
            curr_back_pointer.append(best_prev_state)
        
        deltas.append(curr_delta)
        back_pointers.append(curr_back_pointer)

    # Find the state with the highest probability at the last time step
    last_time_idx = nr_obs_seq - 1
    max_final_prob = -1.0
    best_last_state = 0
    
    for i in range(nr_states):
        if deltas[last_time_idx][i] > max_final_prob:
            max_final_prob = deltas[last_time_idx][i]
            best_last_state = i

    # Reconstruct the path from end to beginning
    path = [0] * nr_obs_seq
    path[last_time_idx] = best_last_state

    for t in range(last_time_idx - 1, -1, -1):
        # Look at the backpointer for the state we chose at t+1
        next_state_in_path = path[t+1]
        path[t] = back_pointers[t+1][next_state_in_path]

    return " ".join(map(str, path))

transition, emission, initial, sequence = parse(file)
print(viterbi_algo(transition, emission, initial, sequence))