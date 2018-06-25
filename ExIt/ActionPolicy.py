
import numpy as np
import random


def get_action_index_exploit(pi, action_indexes):
    """ EXPLOIT.
        Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    return action_indexes[np.argmax(pi)]


def get_action_index_explore(pi, action_indexes):
    """ EXPLORE.
        Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    return np.random.choice(a=action_indexes, size=1, p=pi)[0]


def e_greedy(pi, legal_moves, e):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    best_action = get_action_index_exploit(pi=pi, action_indexes=legal_moves)
    if random.uniform(0, 1) < e:
        return best_action, random.choice(legal_moves)
    return best_action, best_action
