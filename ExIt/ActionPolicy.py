
import numpy as np
import random


def get_action_index_exploit(values, legal_moves):
    """ EXPLOIT.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    return legal_moves[np.argmax(values)]


def get_action_index_explore(values, legal_moves):
    """ EXPLORE.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    return np.random.choice(a=legal_moves, size=1, p=values)[0]


def e_greedy(pi, legal_moves, e):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    best_action = get_action_index_exploit(values=pi, legal_moves=legal_moves)
    if random.uniform(0, 1) < e:
        return best_action, random.choice(legal_moves)
    return best_action, best_action
