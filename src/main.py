

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Matchmaking.Comparison1v1 import compare_ex_it_trained, compare_ex_it_from_scratch
from Matchmaking.EloTournament import start_elo_tournament
from Misc.Debugger import debug_display_win_moves
from Misc.Training import self_play_and_store_versions
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_matches = 1000

iterations = 10
epochs = 100


"""
***** Variable information in the project *****
s   = state
a   = action
fv  = feature vector
v   = predicted value of state
vi  = list of values
pi  = predicted action probability
lm  = legal moves

Q[s]    = Expected Q values from state s
Q[s][a] = Expected Q values when taking action a from state s. 
N[s]    = Number of times state s was visited. 
P[s]    = Predicted P values from state s.
P[s][a] = Predicted P values when taking action a from state s.
V[s]    = Predicted v value of state s.
***********************************************
"""


def main():
    comparison_from_scratch()


def elo_tournament():
    start_elo_tournament(
        players=[NnAlphaBetaPlayer(), NnMinimaxPlayer(), NnMctsPlayer(), RandomPlayer()],
        game_class=TicTacToe,
        trained_iterations=10,
        randomness=True
    )


def train_and_store():
    self_play_and_store_versions(
        ex_it_algorithm=NnMinimaxPlayer().ex_it_algorithm,
        search_time=search_time,
        iterations=iterations,
        epochs=epochs,
        game_class=TicTacToe
    )


def comparison_trained():
    players = [NnAlphaBetaPlayer(), RandomPlayer()]
    compare_ex_it_trained(
        game_class=TicTacToe,
        players=players,
        num_matches=1000,
        randomness=False,
        version=10
    )


def comparison_from_scratch():
    # Run Comparison with several iteration of self-play.
    players = [NnMinimaxPlayer(), RandomPlayer()]

    compare_ex_it_from_scratch(
        game_class=TicTacToe,
        players=players,
        epochs=epochs,
        search_time=search_time,
        num_matches=num_matches,
        iterations=iterations,
        randomness=False
    )


def normal_exit_test():
    # Run One iteration of self.play
    player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(
        epochs=10,
        search_time=search_time
    )

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()')
