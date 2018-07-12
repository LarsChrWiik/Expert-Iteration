

from Games.TicTacToe import TicTacToe
from Players.Players import *
from Matchmaking.Comparison1v1 import compare_ex_it_trained, compare_ex_it_from_scratch
from Matchmaking.EloTournament import start_elo_tournament
from Misc.Debugger import debug_display_win_moves
from Misc.Training import self_play_and_store_versions
from Misc.TrainingTimer import get_seconds
from Misc.TrainingTimer import TrainingTimer
import numpy as np
np.set_printoptions(suppress=True)


"""
***** Variable information in the project *****
s   = state
a   = action
fv  = feature vector
v   = predicted value of state
vi  = list of values
pi  = predicted action probability
lm  = legal moves
t = turn

Q[s]    = Expected Q values from state s
Q[s][a] = Expected Q values when taking action a from state s. 
N[s]    = Number of times state s was visited. 
P[s]    = Predicted P values from state s.
P[s][a] = Predicted P values when taking action a from state s.
V[s]    = Predicted v value of state s.
***********************************************
"""


search_time = get_seconds(ms=50)
num_matches = 1000

training_timer = TrainingTimer(
    time_limit=get_seconds(s=100),
    num_versions=10
)


def main():
    comparison_from_scratch()


def elo_tournament():
    start_elo_tournament(
        players=[NnAlphaBetaPlayer(), NnMctsPlayer(), RandomPlayer()],
        game_class=TicTacToe,
        randomness=True
    )


def train_and_store():
    player = NnMctsPlayer()
    self_play_and_store_versions(
        game_class=TicTacToe,
        ex_it_algorithm=player.ex_it_algorithm,
        search_time=search_time,
        training_timer=training_timer
    )


def comparison_trained():
    players = [NnAlphaBetaPlayer(), RandomPlayer()]
    compare_ex_it_trained(
        game_class=TicTacToe,
        players=players,
        num_matches=num_matches,
        randomness=False,
        version=5
    )


def comparison_from_scratch():
    # Run Comparison with several iteration of self-play.
    players = [NnMctsPlayer(), RandomPlayer()]

    compare_ex_it_from_scratch(
        game_class=TicTacToe,
        players=players,
        search_time=search_time,
        num_matches=num_matches,
        training_timer=training_timer,
        randomness=False
    )


def normal_exit_test():
    # Run One iteration of self.play
    player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(
        training_timer=training_timer,
        search_time=search_time
    )

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()')
