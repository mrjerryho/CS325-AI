#! /usr/bin/env python
import sys
import re

usage = "\nusage:   ./hmm.py weather weather-test1-1000.txt"

# This section defines states and probabilities used by the HMMs.
#
# This assignment has parts concerning two different HMMs: the weather
# model specified in the HMM Tutorial, and the speech HMM for phones from
# problem 15.12 in the Russell and Norvig book.
#
# Note that in the phone HMM, 'none' is always observed in the 'final' state
# and the 'final' state transitions back to 'final' with probability 1.  
# (This fact was not included in the book).

### Weather Model ###

# state map
weatherStateMap   = {'sunny' : 0, 'rainy' : 1, 'foggy' : 2}
weatherStateIndex = {0 : 'sunny', 1 : 'rainy', 2 : 'foggy'}

# observation map
weatherObsMap   = {'no' : 0, 'yes' : 1}
weatherObsIndex = {0 : 'no', 1 : 'yes'}

# prior probability on weather states
# P(sunny) = 0.5  P(rainy) = 0.25  P(foggy) = 0.25
weatherProb = [0.5, 0.25, 0.25]

# transition probabilities
#                    tomorrrow
#    today     sunny  rainy  foggy
#    sunny      0.8    0.05   0.15
#    rainy      0.2    0.6    0.2 
#    foggy      0.2    0.3    0.5
weatherTProb = [ [0.8, 0.05, 0.15], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5] ]

# conditional probabilities of evidence (observations) given weather
#                          sunny  rainy  foggy 
# P(umbrella=no|weather)    0.9    0.2    0.7
# P(umbrella=yes|weather)   0.1    0.8    0.3
weatherEProb = [ [0.9, 0.2, 0.7], [0.1, 0.8, 0.3] ]

# Using the prior probabilities and state map, return:
#     P(state)
def getStatePriorProb(prob, stateMap, state):
    return prob[stateMap[state]]

# Using the transition probabilities and state map, return:
#     P(next state | current state)
def getNextStateProb(tprob, stateMap, current, next):
    return tprob[stateMap[current]][stateMap[next]]

# Using the observation probabilities, state map, and observation map, return:
#     P(obs | state)
def getObservationProb(eprob, stateMap, obsMap, state, obs):
    return eprob[obsMap[obs]][stateMap[state]]

# Normalize a probability distribution
def normalize(pdist):
    s = sum(pdist)
    for i in range(0,len(pdist)):
        pdist[i] = pdist[i] / s
    return pdist


# Filtering.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#         (ie whether the umbrella was seen [yes, no, ...])
#
# Output: The posterior probability distribution over the most recent state
#         given all of the observations: P(X(T-1)|E(0), ..., E(T-1)).
def filter( stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
    """
       You will need to provide the correct implementation.

    """
    # for t in observations
        # step1: elapse time: marginal belief distribution over current state
        # update belief ( p of each state) with current observation'
        # normalize belief distribution


    pdist = []
    prevDist = weatherProb # priors

    for obs in observations:

        pdist = []
        for i in range(0, len(stateMap)):
            pdist.append(0)

        for curState in stateMap:
            i = stateMap[curState]
            for prevState in stateMap:
                j = stateMap[prevState]
                pdist[i] = pdist[i] + prevDist[j]*getNextStateProb(weatherTProb, weatherStateMap, prevState, curState)

            pdist[i] = pdist[i] * getObservationProb(weatherEProb, weatherStateMap, weatherObsMap, curState, obs)
        pdist = normalize(pdist)
        prevDist = pdist
    return pdist

# Prediction.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Output: The posterior probability distribution over the next state
#         given all of the observations: P(X(T)|E(0), ..., E(T-1)).
def predict(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
    """
       You will need to provide the correct implementation.
    """

    curDist = filter(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations)
    nextDist = []

    for i in range(0, len(stateMap)):
        nextDist.append(0)

    for curState in stateMap:
        i = stateMap[curState]
        for prevState in stateMap:
            j = stateMap[prevState]
            nextDist[i] = nextDist[i] + curDist[j] * getNextStateProb(weatherTProb, weatherStateMap, prevState, curState)
    nextDist = normalize(nextDist)
    return nextDist


# Smoothing.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Output: The posterior probability distribution over each state given all
#         of the observations: P(X(k)|E(0), ..., E(T-1) for 0 <= k <= T-1.
#
#         These distributions should be returned as a list of lists. 
def smooth( \
        stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
    """
       You will need to provide the correct implementation.
       The dummy implementation returns a list of probability distrubutions
       that each assign 1 to the first value a state can take and 0 to the rest.
    """
    pdist = []
    for i in range(0,len(stateMap)):
        if (i == 0):
            pdist.append(1)
        else:
            pdist.append(0)
    pdist_list = []
    for k in observations:
        pdist_list.append(pdist)
    return pdist_list


# Viterbi algorithm.
# Input:  The HMM (state and observation maps, and probabilities) 
#         A list of T observations: E(0), E(1), ..., E(T-1)
#      
# Output: A list containing the most likely sequence of states.
#         (ie [sunny, foggy, rainy, sunny, ...])
def viterbi( \
        stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
    """
       You will need to provide the correct implementation.
       The dummy implementation returns a list of identical states.
    """
    seq = []
    for i in observations:
        seq.append(stateIndex[0])
    return seq


# Functions for testing.
# You should not change any of these functions.
def loadData(filename):
    input = open(filename, 'r')
    input.readline()
    data = []
    for i in input.readlines():
        x = i.split()
        y = x[0].split(",")
        data.append(y)
    return data



def test(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, data):
    observations = []
    classes = []
    for c,o in data:
        observations.append(o)
        classes.append(c)
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]
    classes_short = classes[0:n_obs_short]
    print 'Short observation sequence:'
    print '   ', obs_short
    # test filtering
    result_filter = filter( \
        stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
    print '\nFiltering - distribution over most recent state:'
    for i in range(0,len(result_filter)):
        print '   ', stateIndex[i], '%1.3f' % result_filter[i],
    # test prediction
    result_predict = predict( \
        stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
    print '\n\nPrediction - distribution over next state:'
    for i in range(0,len(result_filter)):
        print '   ', stateIndex[i], '%1.3f' % result_predict[i],


if __name__ == '__main__':
    type = None
    filename = None
    if len(sys.argv) > 1:
        type = sys.argv[1]
    if len(sys.argv) > 2:
        filename = sys.argv[2]

    if filename:
        data = loadData(filename)
    else:
        print usage
        exit(0)

    if (type == 'weather'):
        test( \
            weatherStateMap, \
            weatherStateIndex, \
            weatherObsMap, \
            weatherObsIndex, \
            weatherProb, \
            weatherTProb, \
            weatherEProb, \
            data)
    elif (type == 'phone'):
        test( \
            phoneStateMap, \
            phoneStateIndex, \
            phoneObsMap, \
            phoneObsIndex, \
            phoneProb, \
            phoneTProb, \
            phoneEProb, \
            data)
    else:
        print usage
        exit(0)
