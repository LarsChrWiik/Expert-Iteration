
import numpy as np


def bitboard(board, player_index):
    """
    Generate the board feature vector from an np.array.
    This function assumes that the game contains exactly
    one type of piece and exactly two players.

    :param board: 1D np.array.
    :param player_index: int.
    :return: 1D np.array.
    """
    opponent_index = 2 if player_index == 1 else 1
    player = np.where(board == player_index, 1, 0)
    opponent = np.where(board == opponent_index, 1, 0)
    return np.concatenate((player, opponent))
