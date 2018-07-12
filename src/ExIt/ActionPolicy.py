
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
    p = [v/sum(values) for v in values]
    a_on_policy = np.random.choice(a=legal_moves, size=1, p=p)[0]
    return a_on_policy


def e_greedy(xi, lm, e):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and legal_moves corresponds. """
    best_action = exploit_action(xi, lm)
    if random.uniform(0, 1) < e:
        return random.choice(lm), best_action
    return best_action, best_action


def p_proportional(pi, vi, legal_moves):
    """ Assumes that VI has removed moves that are not legal.
        Also assumes that the index of pi and vi corresponds. """
    if len(pi) != len(vi):
        raise Exception("pi and vi do not have equal length, but they should. ")
    pi[np.argmax(vi)] += 0.1
    a_on_policy = explore_action(pi, legal_moves)
    return a_on_policy


# TODO: Not used.
def argmax(array):
    array = np.array(array)
    return np.random.choice(np.flatnonzero(array == array.max()))
