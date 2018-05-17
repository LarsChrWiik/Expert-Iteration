

class Game:
    """
    Abstract class used to ensure that games are compatible
    with Expert Iteration algorithms (ExItAlgorithm).
    """

    def get_state_copy(self):
        raise NotImplementedError("Please Implement this method")

    def get_legal_moves(self, player_index):
        raise NotImplementedError("Please Implement this method")

    def advance(self, player_index, action_index):
        raise NotImplementedError("Please Implement this method")

    def has_game_ended(self):
        raise NotImplementedError("Please Implement this method")

    def update_game_state(self):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vector(self, player_index):
        raise NotImplementedError("Please Implement this method")
