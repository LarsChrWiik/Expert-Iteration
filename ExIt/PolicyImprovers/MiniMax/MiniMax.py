
from ExIt.PolicyImprovers.BasePolicyImprover import BasePolicyImprover
from ExIt.PolicyValuePredictors import BasePolicyValuePredictor
from Games.BaseGame import BaseGame
from ExIt.PolicyImprovers.MiniMax.NodeMiniMax import NodeMiniMax


def minimax(node: NodeMiniMax, should_max, predictor: BasePolicyValuePredictor):
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


class MiniMax(BasePolicyImprover):

    depth = None

    def __init__(self, depth):
        self.depth = depth

    def search_and_store(self, game: BaseGame, predictor: BasePolicyValuePredictor):
        root_node = NodeMiniMax(
            state=game,
            action_index=None,
            original_turn=game.turn,
            depth=self.depth
        )
        root_node.expand_tree()
        minimax(node=root_node, should_max=True, predictor=predictor)
        action_index = root_node.get_best_action_index()
        return action_index
