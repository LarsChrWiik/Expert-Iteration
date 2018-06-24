
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Support.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


def minimax(node: "NodeMiniMax", original_turn):
    """ Minimax implementation.
        This algorithm is compatible with games were the same player
        can make several moves in a row. """
    if node.is_leaf():
        return
    if node.state.turn == original_turn:
        # MAX.
        best_value = float('-inf')
        for child in node.children:
            if child.evaluation is not None:
                minimax(node=child, original_turn=original_turn)
                best_value = max(best_value, child.evaluation)
        node.evaluation = best_value
    else:
        # MIN.
        best_value = float('inf')
        for child in node.children:
            if child.evaluation is not None:
                minimax(node=child, original_turn=original_turn)
                best_value = min(best_value, child.evaluation)
        node.evaluation = best_value


class MiniMax(BaseExpert):
    """ This implementation is limited to Zero-sum,
        two-player deterministic markov games """

    def __init__(self):
        self.timer = Timer()

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        root_node = NodeMiniMax(
            state=state,
            action_index=None,
            original_turn=state.turn,
            depth=0,
            root_node=None,
            predictor=predictor
        )
        self.timer.start_search_timer(search_time=search_time)
        depth = 1
        tmp = False
        while self.timer.have_time_left():
            did_progress = self.iteration(root_node=root_node, to_depth=depth)
            # TODO: Check if this is needed.
            if not did_progress:
                if tmp:
                    break
                # All nodes in this depth has been evaluated.
                depth += 1
                tmp = True
            else:
                tmp = False
        minimax(node=root_node, original_turn=root_node.original_turn)

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
    """ Minimax node that includes an evaluation function, which assumes
        Zero-sum, two-player deterministic markov games """

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

    def default_policy(self):
        """ Evaluate Node """
        self.evaluation = zero_sum_2v2_evaluation(
            state=self.state,
            original_turn=self.original_turn,
            predictor=self.predictor
        )

    def __expand(self):
        """ Expand the tree by adding this new node """
        self.children = []
        possible_actions = self.state.get_legal_moves()
        for action_index in possible_actions:
            state_next = self.state.copy()
            state_next.advance(a=action_index)
            self.children.append(
                NodeMiniMax(
                    state=state_next,
                    action_index=action_index,
                    original_turn=self.original_turn,
                    depth=self.depth + 1,
                    root_node=self.root_node,
                    predictor=self.predictor
                )
            )
