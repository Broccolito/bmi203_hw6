import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
         # Step 1. Initialize variables
        alpha = np.zeros((len(input_observation_states), len(self.hidden_states)))

        first_obs_index = self.observation_states_dict[input_observation_states[0]]
        for s in range(len(self.hidden_states)):
            alpha[0, s] = self.prior_p[s] * self.emission_p[s, first_obs_index]

       
        # Step 2. Calculate probabilities
        for t in range(1, len(input_observation_states)):
            for s in range(len(self.hidden_states)):
                alpha[t, s] = np.sum(alpha[t-1] * self.transition_p[:, s]) * self.emission_p[s, self.observation_states_dict[input_observation_states[t]]]

        # Step 3. Return final probability 
        forward_probability = np.sum(alpha[-1])
        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        T = len(decode_observation_states)
        N = len(self.hidden_states)
        
        viterbi_table = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        
        # Initialization
        first_obs_index = self.observation_states_dict[decode_observation_states[0]]
        for s in range(N):
            viterbi_table[0, s] = self.prior_p[s] * self.emission_p[s, first_obs_index]
            backpointer[0, s] = 0
        
        # Recursion
        for t in range(1, T):
            for s in range(N):
                trans_prob = viterbi_table[t-1] * self.transition_p[:, s]
                max_trans_prob = np.max(trans_prob)
                backpointer[t, s] = np.argmax(trans_prob)
                viterbi_table[t, s] = max_trans_prob * self.emission_p[s, self.observation_states_dict[decode_observation_states[t]]]
        
        # Termination
        best_path_pointer = np.argmax(viterbi_table[-1])
        best_hidden_state_sequence = [self.hidden_states[best_path_pointer]]
        
        # Path backtracking
        for t in range(T-1, 0, -1):
            best_path_pointer = backpointer[t, best_path_pointer]
            best_hidden_state_sequence.insert(0, self.hidden_states[best_path_pointer])
        
        return best_hidden_state_sequence