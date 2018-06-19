
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Support.Timer import Timer
from random import choice as rnd_choice


def minimax(node: "NodeMiniMax", should_max):
    if node.is_leaf():
        return
    elif should_max:
        best_value = float('-inf')
        for child in node.children:
            if child.evaluation is not None:
                minimax(node=child, should_max=False)
                best_value = max(best_value, child.evaluation)
        node.evaluation = best_value
    else:
        best_value = float('inf')
        for child in node.children:
            if child.evaluation is not None:
                minimax(node=child, should_max=True)
                best_value = min(best_value, child.evaluation)
        node.evaluation = best_value


class MiniMax(BaseExpert):

    def __init__(self):
        self.timer = Timer()

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        self.timer.start_search_timer(search_time=search_time)
        root_node = NodeMiniMax(state=state, action_index=None, original_turn=state.turn,
                                depth=0, root_node=None, predictor=predictor)

        depth = 1
        tmp = False
        while self.timer.have_time_left():
            did_progress = self.iteration(root_node=root_node, to_depth=depth)
            if not did_progress:
                if tmp:
                    break
                # All nodes in this depth has been evaluated.
                depth += 1
                tmp = True
            else:
                tmp = False
        minimax(node=root_node, should_max=True)
        v_values, action_indexes = root_node.get_v_actions_and_index()
        return v_values, action_indexes, root_node.evaluation

    @staticmethod
    def iteration(root_node: 'NodeMiniMax', to_depth: int):
        """ One iteration of Minimax """
        node = root_node.tree_policy(to_depth=to_depth)
        did_progress = node is not None
        if did_progress:
            node.default_policy()
        return did_progress


class NodeMiniMax:

    def __init__(self, state: BaseGame, action_index, original_turn, depth, root_node, predictor):
        self.state = state
        self.action_index = action_index
        self.original_turn = original_turn
        self.depth = depth
        self.root_node = root_node if root_node is not None else self
        self.predictor = predictor
        self.children = None
        self.evaluation = None

    def is_leaf(self):
        return self.children is None or len(self.children) == 0

    def get_v_actions_and_index(self):
        """ Return the evaluations for each action index from this state """
        v_values, action_indexes = [], []
        for node in self.children:
            if node.evaluation is not None:
                v_values.append(node.evaluation)
                action_indexes.append(node.action_index)
        return v_values, action_indexes

    def tree_policy(self, to_depth: int):
        """ Find the next unexplored node """
        if self.depth < to_depth:
            if self.children is None:
                self.__expand()
            if len(self.children) == 0:
                return None
            for child in self.children:
                node = child.tree_policy(to_depth=to_depth)
                if node is not None:
                    return node
        elif self.depth == to_depth:
            if self.evaluation is None:
                return self
        return None

    def __expand(self):
        """ Expand the tree by adding this new node """
        self.children = []
        possible_actions = self.state.get_possible_actions()
        for action_index in possible_actions:
            state_next = self.state.copy()
            state_next.advance(action_index=action_index)
            self.children.append(NodeMiniMax(
                state=state_next,
                action_index=action_index,
                original_turn=self.original_turn,
                depth=self.depth + 1,
                root_node=self.root_node,
                predictor=self.predictor
            ))

    def default_policy(self):
        """ Evaluate Node """
        if self.state.is_game_over():
            self.evaluation = self.state.get_result(self.original_turn).value
        else:
            self.evaluation = self.predictor.pred_eval(
                X=self.state.get_feature_vector(self.original_turn)
            )
