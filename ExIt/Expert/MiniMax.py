
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.BaseGame import BaseGame
import random
import numpy as np


def minimax(node: "NodeMiniMax", should_max, predictor: BaseApprentice):
    if node.is_leaf():
        node.evaluate_leaf_node(predictor=predictor)
    elif should_max:
        best_value = -99999.9
        for child in node.children:
            minimax(node=child, should_max=False, predictor=predictor)
            best_value = max(best_value, child.evaluation)
        node.evaluation = best_value
    else:
        best_value = 99999.9
        for child in node.children:
            minimax(node=child, should_max=True, predictor=predictor)
            best_value = min(best_value, child.evaluation)
        node.evaluation = best_value


class MiniMax(BaseExpert):

    def __init__(self, depth):
        self.depth = depth

    def start(self, state: BaseGame, predictor: BaseApprentice):
        root_node = NodeMiniMax(state=state, action_index=None,
                                original_turn=state.turn, depth=self.depth)
        root_node.expand_tree()
        minimax(node=root_node, should_max=True, predictor=predictor)
        # Find the best action index.
        action_index = root_node.get_best_action_index(predictor=predictor)

        pi_update = root_node.get_new_pi()

        return action_index, root_node.evaluation, pi_update


class NodeMiniMax:

    def __init__(self, state: BaseGame, action_index, original_turn, depth):
        self.state = state
        self.action_index = action_index
        self.original_turn = original_turn
        self.depth = depth
        self.children = None
        self.evaluation = None

    def is_leaf(self):
        return self.depth == 0 or self.children is None or len(self.children) == 0

    def expand_tree(self):
        possible_actions = self.state.get_legal_moves(self.state.turn)
        if self.depth > 0:
            self.children = []
            for index, action_index in enumerate(possible_actions):
                # Advance to new state.
                state_next = self.state.get_state_copy()
                state_next.advance(action_index=action_index)
                self.children.append(NodeMiniMax(
                    state=state_next,
                    action_index=action_index,
                    original_turn=self.original_turn,
                    depth=self.depth-1
                ))
            [c.expand_tree() for c in self.children]

    def evaluate_leaf_node(self, predictor: BaseApprentice):
        if self.state.has_finished():
            self.evaluation = self.state.get_reward(self.original_turn)
        else:
            X = self.state.get_feature_vector(self.original_turn)
            self.evaluation = predictor.pred_eval(X=X)

    def get_new_pi(self):
        """
        Generate the updated action probabilities.
        """

        pi_update = np.empty(self.state.num_actions, dtype=float)
        pi_update.fill(-1.0)
        for c in self.children:
            pi_update[c.action_index] = c.evaluation
        #pi_update = NodeMiniMax.normalize(array=pi_update, lower=0, upper=1)
        #s = sum(pi_update)
        #pi_update = np.array([x / s for x in pi_update])

        """
        evaluations = []
        indexes = []
        for c in self.children:
            evaluations.append(c.evaluation)
            indexes.append(c.action_index)
        print("evaluations = " + str(evaluations))

        evaluations = self.normalize(array=evaluations, lower=0.0, upper=1.0)

        pi_update = np.zeros(self.state.num_actions, dtype=float)
        for i, v in enumerate(evaluations):
            pi_update[indexes[i]] = v
        """

        return pi_update

    @staticmethod
    # TODO: check if this is correct.
    def normalize(array, lower, upper):

        # Check of array contain non-unique elements.
        if np.unique(array).size == 1:
            return np.array([upper for _ in range(len(array))])

        diff = upper - lower
        _min = array[0]
        _max = array[0]
        for i, v in enumerate(array):
            if v < _min:
                _min = v
            if v > _max:
                _max = v

        new_array = np.zeros(len(array), dtype=float)

        for i, v in enumerate(array):
            norm = (v - _min) / (_max - _min)
            new_array[i] = (diff * norm) + lower

        return new_array

    def get_best_action_index(self, predictor: BaseApprentice):
        best_values = []
        action_indexes = []
        if self.is_leaf():
            # This is the root node. (return the most probable action).
            # TODO: Should be moved.
            X = self.state.get_feature_vector(self.state.turn)
            legal_moves = self.state.get_legal_moves(self.state.turn)
            action_prob = predictor.pred_prob(X=X)
            for i, v in enumerate(action_prob):
                if i not in legal_moves:
                    action_prob[i] = -1
            return np.argmax(action_prob)
        for c in self.children:
            if len(best_values) == 0:
                best_values.append(c.evaluation)
                action_indexes.append(c.action_index)
            elif c.evaluation == best_values[0]:
                best_values.append(c.evaluation)
                action_indexes.append(c.action_index)
            elif c.evaluation > best_values[0]:
                best_values = [c.evaluation]
                action_indexes = [c.action_index]
        return random.choice(action_indexes)