from hmm import HiddenMarkovModel
import numpy as np



mini_hmm=np.load('./data/mini_weather_hmm.npz')
mini_input=np.load('./data/mini_weather_sequences.npz')

hidden_states = mini_hmm['hidden_states']
observation_states = mini_hmm['observation_states']
prior_p = mini_hmm['prior_p']
transition_p = mini_hmm['transition_p']
emission_p = mini_hmm['emission_p']

hmm_mini = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
observation_sequence = mini_input['observation_state_sequence']
best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

forward_probability = hmm_mini.forward(observation_sequence)
viterbi_sequence = hmm_mini.viterbi(observation_sequence)

print(viterbi_sequence)
print(best_hidden_state_sequence)

assert all(viterbi_sequence == best_hidden_state_sequence)

# # Load the mini weather HMM data from .npz file
# mini_weather_hmm = np.load('./data/mini_weather_hmm.npz', allow_pickle=True)
# list(mini_weather_hmm.keys())

# # Initialize variables from the mini weather HMM data
# hidden_states = mini_weather_hmm['hidden_states']
# observation_states = mini_weather_hmm['observation_states']
# prior_p = mini_weather_hmm['prior_p']
# transition_p = mini_weather_hmm['transition_p']
# emission_p = mini_weather_hmm['emission_p']

# # Create an instance of the HiddenMarkovModel with the mini weather HMM data
# hmm_mini = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

# # Check the loaded data to ensure correctness
# print(hmm_mini)

# # mini_weather_hmm_sequence = np.load('./data/mini_weather_sequences.npz', allow_pickle=True)
# # print(mini_weather_hmm_sequence['observation_state_sequence'])
# # print(mini_weather_hmm_sequence['best_hidden_state_sequence'])


# # Load the full weather HMM data from .npz file
# full_weather_hmm = np.load('./data/full_weather_hmm.npz', allow_pickle=True)
# list(full_weather_hmm.keys())

# # Initialize variables from the mini weather HMM data
# hidden_states = full_weather_hmm['hidden_states']
# observation_states = full_weather_hmm['observation_states']
# prior_p = full_weather_hmm['prior_p']
# transition_p = full_weather_hmm['transition_p']
# emission_p = full_weather_hmm['emission_p']

# # Create an instance of the HiddenMarkovModel with the mini weather HMM data
# hmm_full = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

# # Example observation sequence for the full weather HMM model
# observation_sequence_full_example = np.array(['sunny', 'cloudy', 'rainy', 'snowy'])

# # Compute forward probability for the example observation sequence
# forward_probability_full = hmm_full.forward(observation_sequence_full_example)

# # Compute the most likely sequence of hidden states using the Viterbi algorithm
# viterbi_sequence_full = hmm_full.viterbi(observation_sequence_full_example)

# print((forward_probability_full, viterbi_sequence_full))

# mini_input=np.load('./data/mini_weather_sequences.npz')
# print(mini_input['observation_state_sequence'])