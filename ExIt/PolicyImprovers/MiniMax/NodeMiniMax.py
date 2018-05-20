
from ExIt.PolicyValuePredictors import BasePolicyValuePredictor
from Games.BaseGame import BaseGame
import random


class NodeMiniMax:

    state = None
    action_index = None
    original_turn = None
    depth = None

    children = None
    evaluation = None

    def __init__(self, state: BaseGame, action_index, original_turn, depth):
        self.state = state
        self.action_index = action_index
        self.original_turn = original_turn
        self.depth = depth

    def is_leaf(self):
        return self.depth == 0 or self.children is None or len(self.children) == 0

    def expand_tree(self):
        possible_actions = self.state.get_legal_moves(self.state.turn)
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
        [c.expand_tree() for c in self.children if self.depth > 0]

    def evaluate_leaf_node(self, predictor: BasePolicyValuePredictor):
        if self.state.has_finished():
            self.evaluation = self.state.get_reward(self.original_turn)
        else:
            feature_vector = self.state.get_feature_vector(self.original_turn)
            self.evaluation = predictor.pred_value(feature_vector)

    def get_best_action_index(self):
        best_values = []
        action_indexes = []
        # TODO: If similar value, then chose random NOT first of that value.
        for c in self.children:
            if not best_values:
                best_values.append(c.evaluation)
                action_indexes.append(c.action_index)
            elif c.evaluation == best_values[0]:
                best_values.append(c.evaluation)
                action_indexes.append(c.action_index)
            elif c.evaluation > best_values[0]:
                best_values = [c.evaluation]
                action_indexes = [c.action_index]
            print("action: " + str(c.action_index) + " = eval: " + str(c.evaluation))
        return random.choice(action_indexes)
