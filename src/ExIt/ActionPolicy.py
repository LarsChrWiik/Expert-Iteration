
import numpy as np
import random


def exploit_action(values, legal_moves):
    """ EXPLOIT.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    a_off_policy = legal_moves[np.argmax(values)]
    return a_off_policy


def explore_action(values, legal_moves):
    """ EXPLORE.
        Assumes that 'values' has removed moves that are not legal.
        Also assumes that the index of 'values' and legal_moves corresponds. """
    p = [v/sum(values) for v in values]
    a_on_policy = np.random.choice(a=legal_moves, size=1, p=p)[0]
    return a_on_policy


def e_greedy(pi, legal_moves, e):
    """ Assumes that PI has removed moves that are not legal.
        Also assumes that the index of pi and action_indexes corresponds. """
    best_action = exploit_action(values=pi, legal_moves=legal_moves)
    if random.uniform(0, 1) < e:
        return best_action, random.choice(legal_moves)
    return best_action, best_action


def p_proportional(pi, vi, legal_moves):
    """ Also assumes that the index of pi and action_indexes corresponds. """
    a_off_policy = exploit_action(values=vi, legal_moves=legal_moves)

    pi[a_off_policy] += 0.1
    pi = pi[legal_moves]
    pi = pi / pi.sum()

    a_on_policy = explore_action(pi, legal_moves)

    return a_off_policy, a_on_policy
