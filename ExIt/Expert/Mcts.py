
from ExIt.Expert.BaseExpert import BaseExpert
from Games.BaseGame import BaseGame
from ExIt.Apprentice import BaseApprentice
from random import choice as rnd_choice
from math import sqrt
from math import log as ln
from Support.Timer import Timer


class Mcts(BaseExpert):

    def __init__(self, c):
        self.timer = Timer()
        self.c = c

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        self.timer.start(search_time=search_time)
        root_node = NodeMcts(state=state, action_index=None, original_turn=state.turn,
                             parent=None, root_node=None, c=self.c)
        while self.timer.have_time_left():
            self.iteration(root_node=root_node)
            if root_node.no_search_space:
                break
        action_index = root_node.get_best_action_index()
        return action_index, None

    @staticmethod
    def iteration(root_node: 'NodeMcts'):
        """ One iteration of MCTS """
        node = root_node.tree_policy()
        if root_node.no_search_space:
            return
        r = node.default_policy()
        node.backpropagate(r)


class NodeMcts:

    def __init__(self, state, action_index, original_turn, parent, root_node, c):
        self.state = state
        self.action_index = action_index
        self.original_turn = original_turn
        self.parent = parent
        self.root_node = root_node if root_node is not None else self
        self.children = None
        # Visits.
        self.n = 0
        # Total score.
        self.t = 0
        # Exploration parameter.
        self.c = c
        self.no_search_space = False

    def __expand(self):
        """ Expand the tree by adding this new node """
        self.children = []
        possible_actions = self.state.get_possible_actions()
        for action_index in possible_actions:
            state_next = self.state.copy()
            state_next.advance(action_index=action_index)
            self.children.append(NodeMcts(
                state=state_next,
                action_index=action_index,
                original_turn=self.original_turn,
                parent=self,
                root_node=self.root_node,
                c=self.c
            )
        )

    def tree_policy(self):
        """ Find the next unexplored node """
        if self.children is None:
            if self.n == 0:
                return self
            self.__expand()

        # TODO: This might be stopping too early.
        # Check if there is no more search space.
        if len(self.children) == 0:
            self.root_node.no_search_space = True
            return None

        child = self.get_ucb1_child()
        return child.tree_policy()

    def default_policy(self):
        """ Simulate this expansion with random moved until the game is over """
        state_copy = self.state.copy()
        while not state_copy.is_game_over():
            if state_copy.is_game_over():
                break
            # Make random advance.
            state_copy.advance(rnd_choice(state_copy.get_possible_actions()))
        return state_copy.get_reward(player_index=self.original_turn)

    def backpropagate(self, r):
        """ Update the estimation for all nodes from the node to the root node """
        self.t += r
        self.n += 1
        if self.parent is not None:
            self.parent.backpropagate(r)

    def get_ucb1_child(self):
        max_score = float("-inf")
        best_node = None
        for child in self.children:
            value = child.ucb1()
            if value > max_score:
                max_score = value
                best_node = child
        return best_node

    def ucb1(self):
        n = self.n
        if n <= 0:
            return float("inf")
        t = self.t
        w = t / n
        N = self.root_node.n
        return (w / n) + self.c * sqrt(ln(N) / n)

    def get_best_action_index(self):
        max_t, action_indexes = [], []
        for c in self.children:
            if len(max_t) == 0 or c.t == max_t:
                max_t.append(c.t)
                action_indexes.append(c.action_index)
            elif c.t > max_t[0]:
                max_t, action_indexes = [c.t], [c.action_index]
        # Ensure that there are moves to be made.
        if len(action_indexes) == 0:
            action_indexes = self.state.get_possible_actions()
        return rnd_choice(action_indexes)
