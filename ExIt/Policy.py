
from enum import Enum
import numpy as np
import random


# Used for E-greedy.
exploration_degree = 0.1


def exploit_action(values, lm):
    """ EXPLOIT.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    return lm[np.argmax(values)]


def explore(lm):
    return np.random.choice(lm)


def explore_proportional(values, lm):
    """ EXPLORE.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    p = [v/sum(values) for v in values]
    return np.random.choice(a=lm, size=1, p=p)[0]


def e_greedy(xi, lm, e=None):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and legal_moves corresponds. """
    if e is None:
        e = exploration_degree
    best_action = exploit_action(xi, lm)
    if random.uniform(0, 1) < e:
        return random.choice(lm)
    return best_action


def e_greedy_action(a, a_best, e=None):
    if e is None:
        e = exploration_degree
    if random.uniform(0, 1) < e:
        return a
    return a_best


def explore_proportional_with_guidance(values, guide_values, lm, guide_percent=0.1):
    """ Assumes that the inputs corresponds to each other and to legal moves """
    if len(values) != len(guide_values) or len(values) != len(lm):
        raise Exception("Inputs have different length. They should correspond. ")
    values[np.argmax(guide_values)] += guide_percent
    return explore_proportional(values, lm)


def vi_proportional(vi, lm):
    """ Assumes that VI has removed moves that are not legal.
        NB: This assumes a zero sum game with rewards between -1 and 1. """
    vi = [x + 1 for x in vi]
    s = sum(vi)
    if s == 0:
        vi2 = [1 / len(vi) for _ in vi]
    else:
        vi2 = [x / s for x in vi]
    return explore_proportional(vi2, lm)


# TODO: Not used.
def argmax(array):
    """ Argmax implementation that chooses randomly
        when multiple indexes contain the highest value. """
    array = np.array(array)
    return np.random.choice(np.flatnonzero(array == array.max()))


class Policy(Enum):
    ON = "ON"
    OFF = "OFF"
