
import numpy as np
import random


# 1.0 = always explore. 0.0 = always exploit.
exploration_degree = 0.1


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


def e_greedy(pi, legal_moves):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    if random.uniform(0, 1) < exploration_degree:
        return random.choice(legal_moves)
    return get_action_index_exploit(pi=pi, action_indexes=legal_moves)
