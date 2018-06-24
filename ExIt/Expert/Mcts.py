
from ExIt.Expert.BaseExpert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.Apprentice import BaseApprentice
from math import sqrt
from Support.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


class Mcts(BaseExpert):
    """ This MCTS implementation is limited to Zero-sum,
        two-player deterministic markov games """

    def __init__(self, c):
        self.timer = Timer()
        self.c = c

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        root_node = NodeMcts(
            state=state,
            a=None,
            original_turn=state.turn,
            predictor=predictor,
            parent=None,
            root_node=None,
            c=self.c
        )
        self.timer.start_search_timer(search_time=search_time)
        while self.timer.have_time_left():
            root_node.tree_policy()

        v_values = [node.q for node in root_node.children]
        action_indexes = [node.a for node in root_node.children]
        return v_values, action_indexes, root_node.q


class NodeMcts:
    """ MCTS node that includes an evaluation function, which assumes
        Zero-sum, two-player deterministic markov games """

    def __init__(self, state, a, original_turn, predictor, parent, root_node, c):
        self.state = state
        # action_index that lead to this state.
        self.a = a
        self.original_turn = original_turn
        self.predictor = predictor
        self.parent = parent
        self.root_node = root_node if root_node is not None else self
        self.children = None
        # Q value for this state.
        self.q = 0
        self.v = zero_sum_2v2_evaluation(
            state=self.state,
            original_turn=self.original_turn,
            predictor=self.predictor
        )
        # Pi from this state.
        self.p = self.predictor.pred_prob(X=self.state.get_feature_vector())
        # State action visit count.
        self.n = 0
        # Exploration parameter.
        self.c = c

    def tree_policy(self):
        if self.children is None:
            self.__expand()
            self.backpropagate(self.v)

        if len(self.children) == 0:
            # This node is a leaf node.
            self.backpropagate(self.v)
        else:
            self.get_ucb_child().tree_policy()

    def get_ucb_child(self):
        """ Finds the child this the highest UCB value """
        max_score = float("-inf")
        best_node = None
        for child in self.children:
            value = child.ucb()
            if value > max_score:
                max_score = value
                best_node = child
        return best_node

    def ucb(self):
        """ Calculates Upper Confident Bound for this child """
        nsb = [c.n for c in self.parent.children]
        return self.q + self.c * self.parent.p[self.a] * sqrt(sum(nsb)) / (1 + self.n)

    def backpropagate(self, v):
        """ Update the estimation for all nodes from the node to the root node """
        self.n += 1
        self.q = (self.n * self.q + v) / (self.n + 1)
        if self.parent is not None:
            self.parent.backpropagate(v)

    def __expand(self):
        """ Expand the tree by adding this new node """
        self.children = []
        possible_actions = self.state.get_legal_moves()
        for a in possible_actions:
            state_next = self.state.copy()
            state_next.advance(a=a)
            self.children.append(
                NodeMcts(
                    state=state_next,
                    a=a,
                    original_turn=self.original_turn,
                    predictor=self.predictor,
                    parent=self,
                    root_node=self.root_node,
                    c=self.c
                )
            )
