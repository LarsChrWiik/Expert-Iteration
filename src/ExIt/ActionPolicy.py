
import numpy as np
import random


def exploit_action(values, legal_moves):
    """ EXPLOIT.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    return legal_moves[np.argmax(values)]


def explore_action(values, legal_moves):
    """ EXPLORE.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    return np.random.choice(a=legal_moves, size=1, p=values)[0]


def e_greedy(p, legal_moves, e):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    best_action = exploit_action(values=p, legal_moves=legal_moves)
    if random.uniform(0, 1) < e:
        return best_action, random.choice(legal_moves)
    return best_action, best_action


def p_proportional(p, v, legal_moves):
    """ Assumes that p indexes corresponds to action_indexes. """
    best_action = exploit_action(values=v, legal_moves=legal_moves)

    p[best_action] += 0.1
    p = p[legal_moves]
    p = p / p.sum()

    action_taken = explore_action(p, legal_moves)

    return best_action, action_taken
