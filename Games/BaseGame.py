

class BaseGame:
    """
    Abstract class used to ensure that games are compatible
    with Expert Iteration algorithms (ExItAlgorithm).
    Every subclass of Game should be a perfect information game.
    """

    board = None
    player_count = None

    # Indicates the winner of the game. (Index of the winning player). (-1 = draw)
    winner = None

    # 0 = player1, 1 = player2. (Index of the winning player).
    turn = None

    def has_game_ended(self):
        return self.winner is not None

    def is_game_draw(self):
        raise NotImplementedError("Please Implement this method")

    def next_turn(self):
        raise NotImplementedError("Please Implement this method")

    def get_state_copy(self):
        raise NotImplementedError("Please Implement this method")

    def get_legal_moves(self, player_index):
        raise NotImplementedError("Please Implement this method")

    def advance(self, player_index, action_index):
        raise NotImplementedError("Please Implement this method")

    def update_game_state(self):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vector(self, player_index):
        raise NotImplementedError("Please Implement this method")

    def display(self):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def __rep_value_to_p_index(rep_value):
        """
        Convert Representation Value to player index.

        :param rep_value: int, representation value for a game piece.
        :return: int.
        """
        raise NotImplementedError("Please Implement this method")
