

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Support.Debugger import TicTacToeDebugger
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_iteration = 10
num_train_epoch = 10000
num_matches = 1000

debugger = TicTacToeDebugger()


def main():
    # Run Comparison with several iteration of self-play.
    """
    m = Matchmaking(game_class=TicTacToe,
                    players=[NnMinimaxPlayer(), RandomPlayer()])

    m.compare_ex_it(num_train_epoch=num_train_epoch,
                    search_time=search_time,
                    num_matches=num_matches,
                    num_iteration=num_iteration,
                    file_name="NnMinimaxPlayer_vs_RandomPlayer")
    """


    # Run One iteration of self.play
    player = NnMinimaxPlayer()
    #player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(num_iteration=10000,
                       randomness=True,
                       search_time=search_time)


    # Display the agents calculations in predefined scenarios.
    debugger.debug_display_win_moves(player)


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
    #plot_test()
