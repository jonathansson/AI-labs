#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        # Store observation history for all fish
        self.observations = []
        # Track which fish IDs we have already guessed
        self.guessed_fishes = set()
        # Store trained HMM models (A, B, pi) for each species
        self.models = [None] * N_SPECIES
        
        # Store pre-calculated log-probabilities of emissions.
        self.log_emissions = [None] * N_SPECIES
        
        # Track performance stats
        self.correct_cnt = 0
        
        # We use N=1 (1 hidden state)
        self.N = 1      
        self.M = N_EMISSIONS 
        # Wait tick to ensure we collect enough data before guessing
        self.start_tick = 95 

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        # 1. Update observations
        if step == 1:
            self.observations = [[o] for o in observations]
        else:
            for i, o in enumerate(observations):
                self.observations[i].append(o)

        # Wait until we have enough data points
        if step < self.start_tick:
            return None
        
        # 2. Match unguessed fish against known models
        best_fish = -1
        best_species = -1
        best_score = -np.inf

        # Identify fish we havent guessed yet
        unguessed_ids = [i for i in range(len(self.observations)) if i not in self.guessed_fishes]
        
        if not unguessed_ids:
            return None

        for fish_id in unguessed_ids:
            full_obs = self.observations[fish_id]
            
            # Limit history to the last 180 steps.
            # This prevents timeout errors on long test cases.
            if len(full_obs) > 180:
                obs_list = full_obs[-180:]
            else:
                obs_list = full_obs
            
            # Convert to numpy array once for fast indexing
            obs_arr = np.array(obs_list, dtype=int)
            
            for species_id, log_B in enumerate(self.log_emissions):
                if log_B is not None:
                    # Since N=1, the probability of a sequence is just the product of emission probabilities.
                    # In log-space, this product becomes a sum.
                    score = np.sum(log_B[obs_arr])
                    
                    # Normalize by length to compare fish fairly
                    avg_score = score / len(obs_list)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_fish = fish_id
                        best_species = species_id

        # 3. Decision
        # Threshold -2.0 is slightly better than random guessing
        if best_fish != -1 and best_score > -2.0:
            return (best_fish, best_species)
        
        # 4. Reveal strategy
        # If no good match, sacrifice a guess to learn a new model
        target_fish = unguessed_ids[0]
        
        # Prioritize learning species we haven't found yet
        for s in range(N_SPECIES):
            if self.models[s] is None:
                return (target_fish, s)
        
        # Fallback: Random guess
        return (target_fish, random.randint(0, N_SPECIES - 1))

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        self.guessed_fishes.add(fish_id)
        
        if correct:
            self.correct_cnt += 1
        
        # Prepare data for training
        obs = np.array(self.observations[fish_id])
        
        # Initialize HMM parameters with random noise.
        # Stamp (section 7), random initialization is critical 
        # to avoid getting stuck in local maxima.
        A = np.random.rand(self.N, self.N) + 0.1
        A /= A.sum(axis=1, keepdims=True)
        
        B = np.random.rand(self.N, self.M) + 0.1
        B /= B.sum(axis=1, keepdims=True)
        
        pi = np.random.rand(self.N) + 0.1
        pi /= pi.sum()

        # Train the model using the Baum-Welch algorithm (Problem 3 in Stamp)
        final_A, final_B, final_pi = baum_welch(obs, A, B, pi, self.N, self.M)
        
        self.models[true_type] = (final_A, final_B, final_pi)
        
        # We add a small epsilon to avoid log(0) errors
        self.log_emissions[true_type] = np.log(final_B[0] + 1e-10)


def baum_welch(obs, A, B, pi, N, M, iterations=10):
    """
    Baum-Welch Algorithm implementation
    """
    T = len(obs)
    
    for _ in range(iterations):
        # 1. Forward Pass (Alpha)
        # Corresponds to Stamp Section 4.1 and Section 6 (scaling)
        alpha = np.zeros((T, N))
        c = np.zeros(T) # Scaling factors
        
        # Initialization
        alpha[0] = pi * B[:, obs[0]]
        # Calculate scaling factor c0 as described in Stamp section 6
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-100)
        alpha[0] *= c[0]
        
        # Induction
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
            # Scale alpha to prevent underflow (Stamp Eq 13)
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-100)
            alpha[t] *= c[t]

        # 2. Backward Pass (Beta)
        # Corresponds to Stamp Section 4.2 and Section 6 (scaling)
        beta = np.zeros((T, N))
        beta[T-1] = c[T-1] # Initialize with last scaling factor

        # Induction (backwards)
        for t in range(T-2, -1, -1):
            obs_prob = B[:, obs[t+1]] * beta[t+1]
            # Beta uses the same scaling factors c as Alpha (stamp Section 6)
            beta[t] = (A @ obs_prob) * c[t]

        # 3. Compute Gamma & Xi
        # Corresponds to Stamp Section 4.3 (re-estimation).
        
        # Gamma: Probability of being in state i at time t
        gamma = alpha * beta
        gamma /= (np.sum(gamma, axis=1, keepdims=True) + 1e-100)

        # Xi: Probability of transition from i to j at time t
        xi_sum = np.zeros((N, N))
        
        for t in range(T-1):
            # Compute xi for each transition
            term1 = alpha[t][:, np.newaxis]
            term2 = A
            term3 = B[:, obs[t+1]] * beta[t+1]
            xi = term1 * term2 * term3
            xi_sum += xi

        # 4. Update Parameters
        # Re-estimation formulas from Stamp section 4.3
        epsilon = 1e-4 # Smoothing to prevent zero probabilities
        
        pi = gamma[0]

        # re-estimate A (transition matrix)
        gamma_sum_A = np.sum(gamma[:-1], axis=0)[:, np.newaxis]
        A = (xi_sum + epsilon) / (gamma_sum_A + (N * epsilon))

        # re-estimate B (emission matrix)
        gamma_sum_B = np.sum(gamma, axis=0)[:, np.newaxis]
        
        obs_onehot = np.zeros((T, M))
        obs_onehot[np.arange(T), obs] = 1
        B_numerator = gamma.T @ obs_onehot
        
        B = (B_numerator + epsilon) / (gamma_sum_B + (M * epsilon))

    return A, B, pi