

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Support.Debugger import TicTacToeDebugger
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_iteration = 10
num_train_epoch = 10000
num_matches = 1000

debugger = TicTacToeDebugger()


def main():
    """
    m = Matchmaking(game_class=TicTacToe,
                    players=[NnMinimaxPlayer(), RandomPlayer()])

    m.compare_ex_it(num_train_epoch=num_train_epoch,
                    search_time=search_time,
                    num_matches=num_matches,
                    num_iteration=num_iteration,
                    file_name="NnMinimaxPlayer_vs_RandomPlayer")
    """

    player = NnMinimaxPlayer()
    #player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(num_iteration=10000,
                       randomness=True,
                       search_time=search_time)

    # Display the agents calculations.
    debugger.debug_display_win_moves(player)


# TODO: Fix metafile.
import csv
def plot_test(metafile=None, file=None):

    with open('./Statistics/2018-06-18___02-20-59/0.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [[float(row["win"]), float(row["loss"]), float(row["draw"])] for row in reader]

        print(data)
        iterations = 1000
        wins, loses, draws = get_statistics(data=data, iterations=iterations)
        plt.plot(wins, label="Wins")
        plt.plot(loses, label="Loses")
        plt.plot(draws, label="Draws")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)
        plt.axis(ymin=0, ymax=1.0)
        plt.show()


def get_statistics(data, iterations):
    wins = []
    loses = []
    draws = []
    for v in data:
        wins.append(v[0] / iterations)
        loses.append(v[1] / iterations)
        draws.append(v[2] / iterations)
    return wins, loses, draws


if __name__ == "__main__":
    main()
    #cProfile.run('main()')
    #plot_test()
