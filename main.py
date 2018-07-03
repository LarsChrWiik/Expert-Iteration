

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Support.Debugger import debug_display_win_moves
from Matchmaking.Matchmaking import Matchmaking
from Support.Plotter import plot_result
from Training import self_play_and_store_versions
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_matches = 1000

iterations = 10
epochs = 10


def main():
    train()


def train():
    self_play_and_store_versions(
        ex_it_algorithm=NnAlphaBetaPlayer().ex_it_algorithm,
        search_time=search_time,
        iterations=iterations,
        epochs=epochs,
        game_class=TicTacToe
    )


def normal_test():
    # Run One iteration of self.play
    player = NnAlphaBetaPlayer()

    #player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(
        epochs=1,
        search_time=search_time
    )

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


def comparison():
    # Run Comparison with several iteration of self-play.
    Matchmaking(
        game_class=TicTacToe,
        players=[NnMctsPlayer(), RandomPlayer()]
    ).compare_ex_it(
        num_train_epoch=epochs,
        search_time=search_time,
        num_matches=num_matches,
        num_iteration=iterations,
        randomness=False
    )


def plot():
    folder = '2018-06-20___00-39-18'
    filename = "0"
    plot_result(folder=folder, filename=filename)


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
