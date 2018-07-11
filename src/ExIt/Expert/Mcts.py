
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Misc.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation
from math import sqrt
from random import shuffle


class Mcts(BaseExpert):

    def __init__(self, c):
        self.timer = Timer()
        # Exploration parameter in UCB.
        self.c = c

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        # Expected Q values from state s.       Q[s]   or   Q[s][a]
        Q = {}
        # Number of times state s was visited.  N[s]
        N = {}
        # Predicted P values from state s.      P[s]   or   P[s][a]
        P = {}
        # Predicted v value of state s.         V[s]
        V = {}

        original_turn = state.turn

        def mcts_search(state, is_root=False):

            fv = state.get_feature_vector()
            legal_moves = state.get_legal_moves()
            s = tuple(fv)

            # When unexplored child - predict and store info from this state.
            if s not in P:
                P[s] = predictor.pred_p(X=fv)
                V[s] = zero_sum_2v2_evaluation(state, original_turn, predictor)
                N[s] = [0 for _ in range(state.num_actions)]
                Q[s] = [0 for _ in range(state.num_actions)]
                return V[s]

            # Return v value if state is game over.
            if state.is_game_over():
                return V[s]

            # Find action that maximizes Upper Confidence Bound (UCB).
            u_max = -float("inf")
            a_best = -1
            a_shuffled = list(enumerate(legal_moves))
            shuffle(a_shuffled)
            for i, a in a_shuffled:
                if N[s][a] == 0:
                    # Choose this action if it has not been tried. 
                    a_best = a
                    break
                else:
                    u = Q[s][a] + self.c * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
                if u > u_max:
                    u_max = u
                    a_best = a
            # Action that maximizes UCB.
            a = a_best

            # Recursive call to find the v value to backpropagate.
            next_state = state.copy()
            next_state.advance(a)
            v = mcts_search(next_state)

            # Backpropagation step - update Q and N.
            Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
            N[s][a] += 1

            return v

        """ ***** SEARCH CODE ***** """

        self.timer.start_search_timer(search_time)
        while self.timer.have_time_left():
            mcts_search(state, is_root=True)

        # Get V values and action indexes of legal moves.
        legal_moves = state.get_legal_moves()
        s = tuple(state.get_feature_vector())
        # TODO: change to N[]
        v_values = [n for i, n in enumerate(N[s]) if i in legal_moves]
        v_root = None

        return v_values, legal_moves, v_root
