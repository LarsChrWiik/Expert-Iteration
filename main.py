

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Support.Debugger import debug_display_win_moves
from Matchmaking.Matchmaking import Matchmaking
from Support.Plotter import plot_result
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.2
num_iteration = 10
num_train_epoch = 10000
num_matches = 1000


def main():
    normal_test()


def normal_test():
    # Run One iteration of self.play
    player = NnMinimaxPlayer()
    # player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(
        num_iteration=1000,
        search_time=search_time
    )

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


def comparison():
    # Run Comparison with several iteration of self-play.
    Matchmaking(
        game_class=TicTacToe,
        players=[NnMinimaxPlayer(), RandomPlayer()]
    ).compare_ex_it(
        num_train_epoch=num_train_epoch,
        search_time=search_time,
        num_matches=num_matches,
        num_iteration=num_iteration,
        file_name="NnMinimaxPlayer_vs_RandomPlayer",
        randomness=True
    )


def plot():
    folder = '2018-06-18___02-20-59'
    filename = "0"
    plot_result(folder=folder, filename=filename)


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
    #plot()
