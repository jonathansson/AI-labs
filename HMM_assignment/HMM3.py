import sys
import math
import time

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

def baum_welch(transition, emission, initial, sequence):
    N = int(transition[0]) # Number of states
    M = int(emission[1]) # Number of emissions
    T = int(sequence[0]) # Length of observation sequence
    obs = list(map(int, sequence[1:])) # The actual observations

    A = []
    for i in range(N):
        A.append(transition[2 + i*N : 2 + (i+1)*N])
    
    B = []
    for i in range(N):
        B.append(emission[2 + i*M : 2 + (i+1)*M])
    
    pi = initial[2:]

    # Training control
    max_iters = 100
    eps = 1e-6
    prev_logprob = float("-inf")
    
    start_time = time.time()

    for iteration in range(max_iters):
        
        if time.time() - start_time > 0.9:
            break
            
        # The alpha pass (forward) with scaling
        scal = [0.0] * T # Scaling factors
        alpha_list = [[0.0] * N for _ in range(T)]

        # Alpha initialization (t=0)
        c0 = 0.0
        for i in range(N):
            alpha_list[0][i] = pi[i] * B[i][obs[0]]
            c0 += alpha_list[0][i]
        
        # safe scale alpha[0]
        if c0 < eps:
            c0 = 1.0
        scal[0] = 1.0 / c0
        for i in range(N):
            alpha_list[0][i] *= scal[0]

        # Alpha recursion (t=1 to T-1)
        for t in range(1, T):
            ct = 0.0
            for i in range(N):
                sum_prev = 0.0
                for j in range(N):
                    sum_prev += alpha_list[t-1][j] * A[j][i]
                alpha_list[t][i] = sum_prev * B[i][obs[t]]
                ct += alpha_list[t][i]
            
            # Scale alpha[t]
            if ct < eps:
                ct = 1.0
            scal[t] = 1.0 / ct
            for i in range(N):
                alpha_list[t][i] *= scal[t]

        # The beta pass (backward) with scaling 
        beta_list = [[0.0] * N for _ in range(T)]

        # Beta initialization (t=T-1)
        for i in range(N):
            beta_list[T-1][i] = scal[T-1] # scaled by c[T-1]

        # Beta recursion (t=T-2 down to 0)
        for t in range(T-2, -1, -1):
            for i in range(N):
                sum_next = 0.0
                for j in range(N):
                    sum_next += A[i][j] * B[j][obs[t+1]] * beta_list[t+1][j]
                beta_list[t][i] = sum_next * scal[t] # scale by c[t]

        # Accumulate di-gammas and gammas for re-estimation
        new_A = [[0.0] * N for _ in range(N)]
        new_B = [[0.0] * M for _ in range(N)]

        # t = 0..T-2: xi and gamma accumulations
        for t in range(T-1):
            # Calculate the total sum of unnormalized xi
            denom = 0.0
            xi_raw = [[0.0] * N for _ in range(N)]
            
            for i in range(N):
                for j in range(N):
                    xi_raw[i][j] = alpha_list[t][i] * A[i][j] * B[j][obs[t+1]] * beta_list[t+1][j]
                    denom += xi_raw[i][j]

            # Accumulate normalized values
            if denom > eps: 
                for i in range(N):
                    normalized_gamma_i = 0.0
                    for j in range(N):
                        
                        normalized_xi = xi_raw[i][j] / denom
                        
                        # Accumulate A (di-gamma)
                        new_A[i][j] += normalized_xi
                        
                        normalized_gamma_i += normalized_xi
                    
                    # Accumulate B (gamma)
                    new_B[i][obs[t]] += normalized_gamma_i

        # last time step gamma_{T-1}(i) must use alpha * beta (scaled)
        denom = 0.0
        for i in range(N):
            denom += alpha_list[T-1][i] * beta_list[T-1][i]
            
        if denom > eps:
            for i in range(N):
                 gamma_T_i = alpha_list[T-1][i] * beta_list[T-1][i] / denom
                 new_B[i][obs[T-1]] += gamma_T_i


        # Update model parameters:

        # Update pi using gamma_0 (normalized)
        pi_denom = 0.0
        for i in range(N):
            pi_denom += alpha_list[0][i] * beta_list[0][i]
            
        if pi_denom > eps:
            for i in range(N):
                pi[i] = (alpha_list[0][i] * beta_list[0][i]) / pi_denom

        # Update A 
        for i in range(N):
            denom_a = sum(new_A[i])
            if denom_a > eps: # Only update if state i was visited
                for j in range(N):
                    A[i][j] = new_A[i][j] / denom_a
            # Else: A[i] remains unchanged

        # Update B 
        for i in range(N):
            denom_b = sum(new_B[i])
            if denom_b > eps: # Only update if state i was visited
                for k in range(M):
                    B[i][k] = new_B[i][k] / denom_b

        # compute log-likelihood using scaling factors: log P = -sum log(c[t])
        current_p = 0.0
        for t in range(T):
            if scal[t] > 0:
                current_p -= math.log(scal[t])

        # convergence check
        old_p = prev_logprob
        if iteration > 0 and abs(current_p - old_p) < eps:
            break
        prev_logprob = current_p

    # Convert 2D matrices back to the requested string format
    
    # Format A
    res_A = [str(N), str(N)]
    for row in A:
        for val in row:
            res_A.append(f"{val:.6f}") 
    print(" ".join(res_A))

    # Format B
    res_B = [str(N), str(M)]
    for row in B:
        for val in row:
            res_B.append(f"{val:.6f}")
    print(" ".join(res_B))

transition, emission, initial, sequence = parse(file)
baum_welch(transition, emission, initial, sequence)