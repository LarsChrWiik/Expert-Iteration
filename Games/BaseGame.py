

class BaseGame:
    """
    Abstract class used to ensure that games are compatible
    with Expert Iteration algorithms (ExItAlgorithm).
    Every subclass of Game should be a perfect information game.
    """

    win_reward = 1
    lose_reward = -1
    draw_reward = 0

    board = None
    num_players = None
    num_actions = None
    num_rotations = None
    fv_size = None

    # Indicates the winner of the game. (Index of the winning player). (-1 = draw)
    winner = None

    # 0 = player1, 1 = player2. (Index of the winning player).
    turn = None

    def is_game_over(self):
        return self.winner is not None

    def get_reward(self, player_index):
        if player_index == self.winner:
            return self.win_reward
        elif self.is_draw():
            return self.draw_reward
        return self.lose_reward

    def is_draw(self):
        raise NotImplementedError("Please Implement this method")

    def next_turn(self):
        raise NotImplementedError("Please Implement this method")

    def copy(self):
        raise NotImplementedError("Please Implement this method")

    def init_new_game(self):
        raise NotImplementedError("Please Implement this method")

    def get_possible_actions(self):
        raise NotImplementedError("Please Implement this method")

    def advance(self, action_index):
        raise NotImplementedError("Please Implement this method")

    def update_game_state(self):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vector(self, player_index):
        raise NotImplementedError("Please Implement this method")

    def display(self):
        raise NotImplementedError("Please Implement this method")

    def rotate_fv(self, fv):
        raise NotImplementedError("Please Implement this method")

    def rotate_pi(self, pi):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def __rep_value_to_p_index(rep_value):
        """
        Convert Representation Value to player index.

        :param rep_value: int, representation value for a game piece.
        :return: int.
        """
        raise NotImplementedError("Please Implement this method")
