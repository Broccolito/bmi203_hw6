import pytest
from hmm import HiddenMarkovModel
import numpy as np

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

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

    assert all(viterbi_sequence == best_hidden_state_sequence)
   
    pass



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    hidden_states = full_hmm['hidden_states']
    observation_states = full_hmm['observation_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    hmm_full = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    observation_sequence = full_input['observation_state_sequence']
    best_hidden_state_sequence = full_input['best_hidden_state_sequence']

    forward_probability = hmm_full.forward(observation_sequence)
    viterbi_sequence = hmm_full.viterbi(observation_sequence)

    assert all(viterbi_sequence == best_hidden_state_sequence)


    pass













