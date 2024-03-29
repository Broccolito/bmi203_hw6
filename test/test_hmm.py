import pytest
from hmm import HiddenMarkovModel
import numpy as np
from hmmlearn import hmm
import math

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

    # Calculate forward probability using the Hmmlearn module
    n_components = len(hidden_states)
    model = hmm.CategoricalHMM(n_components=n_components)
    model.startprob_ = prior_p
    model.transmat_ = transition_p
    model.emissionprob_ = emission_p
    observation_sequence = np.array(observation_sequence)
    # Map your observations to integer values
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in observation_sequence]).reshape(-1, 1)
    logscore_hmmlearn = model.score(observed_sequence)
    logscore_hmm = np.log(forward_probability)

    assert all(viterbi_sequence == best_hidden_state_sequence)
    assert math.isclose(logscore_hmmlearn, logscore_hmm, rel_tol=1e-5)

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

    # Calculate forward probability using the Hmmlearn module
    n_components = len(hidden_states)
    model = hmm.CategoricalHMM(n_components=n_components)
    model.startprob_ = prior_p
    model.transmat_ = transition_p
    model.emissionprob_ = emission_p
    observation_sequence = np.array(observation_sequence)
    # Map your observations to integer values
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in observation_sequence]).reshape(-1, 1)
    logscore_hmmlearn = model.score(observed_sequence)
    logscore_hmm = np.log(forward_probability)

    assert all(viterbi_sequence == best_hidden_state_sequence)
    assert math.isclose(logscore_hmmlearn, logscore_hmm, rel_tol=1e-5)

    pass


def test_empty_observation_sequence():
    """
    Test handling of empty observation sequence.
    """
    # Using the mini_weather_hmm dataset
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    hmm_mini = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    observation_sequence = np.array([])  # Empty sequence

    # Expecting forward to return log(0) or an exception
    with pytest.raises(ValueError):
        forward_probability = hmm_mini.forward(observation_sequence)

    # Expecting Viterbi to return empty sequence or an exception
    with pytest.raises(ValueError):
        viterbi_sequence = hmm_mini.viterbi(observation_sequence)


def test_single_observation_sequence():
    """
    Test handling of single observation sequence.
    """
    # Assuming mini_weather_hmm for simplicity
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    observation_sequence = np.array(['rainy'])  # Single observation

    # Setup similar to above, using mini_weather_hmm
    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    hmm_mini = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    
    # Run forward algorithm on single observation sequence
    forward_probability = hmm_mini.forward(observation_sequence)
    assert forward_probability > 0, "Forward probability should be positive for a valid observation sequence"

    # Run Viterbi algorithm on single observation sequence
    viterbi_sequence = hmm_mini.viterbi(observation_sequence)
    assert len(viterbi_sequence) == 1, "Viterbi sequence should contain exactly one state for a single observation"
    assert viterbi_sequence[0] in hidden_states, "Viterbi sequence state should be a valid hidden state"











