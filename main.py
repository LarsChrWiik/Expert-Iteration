

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Support.Debugger import debug_display_win_moves, debug_display_general_moves
from Matchmaking.Matchmaking import Matchmaking
from Support.Plotter import plot_result
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_iteration = 100
num_train_epoch = 100
num_matches = 1000


def main():
    comparison()


def normal_test():
    # Run One iteration of self.play
    #player = NnMinimaxPlayer()
    player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(
        num_iteration=1,
        search_time=search_time
    )

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


def test_easy_learning():
    player = NnMinimaxPlayer()
    s_array = [[1, 0, 0, 0, 0, 0, 0, 0, 0,   0, 1, 0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 1, 0, 0, 0, 0,   1, 0, 0, 0, 0, 0, 0, 0, 1]]

    v_values = [[0], [1], [1], [1], [1], [0], [1]]

    pi_values = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    player.get_apprentice_DO_NOT_USE().init_model(
        input_fv_size=len(s_array[0]),
        pi_size=len(pi_values[0])
    )

    for _ in range(1000):
        player.get_apprentice_DO_NOT_USE().train(
            X=s_array,
            Y_pi=pi_values,
            Y_v=v_values
        )
    debug_display_general_moves(player)


def comparison():
    # Run Comparison with several iteration of self-play.
    Matchmaking(
        game_class=TicTacToe,
        players=[NnMctsPlayer(), NnMinimaxPlayer()]
    ).compare_ex_it(
        num_train_epoch=num_train_epoch,
        search_time=search_time,
        num_matches=num_matches,
        num_iteration=num_iteration,
        randomness=True
    )


def plot():
    folder = '2018-06-20___00-39-18'
    filename = "0"
    plot_result(folder=folder, filename=filename)


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
