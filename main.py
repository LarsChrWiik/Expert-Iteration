

from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from Players.Players import *
from Matchmaking.EloTournament import start_elo_tournament
from Misc.Training import self_play_and_store_versions
from Misc.TrainingTimer import get_seconds
from Misc.TrainingTimer import TrainingTimer
from ExIt.Policy import Policy
from Misc.PlayGameCLI import play, play_player
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


# ********** Run info START **********

# Game.
game_class = ConnectFour

# Players to compare.
players = [
    RandomPlayer(),
    NnAlphaBetaPlayer(),
    NnMctsPlayer()
]
# Search time for each player.
search_time = get_seconds(s=1.0)

# Total time for each player to self-train.
time_limit = get_seconds(h=1)
# Number of versions to be trained.
num_versions = 10
# Timer. NB: Each version is trained for time_limit / num_versions time).
training_timer = TrainingTimer(time_limit, num_versions)

# Number of matches to compare the players. This is used to calculate Elo.
# More matches = more certain of elo scores.
num_elo_matches = 10000
# Should some random moves be added in the tournament to generate new states.
match_randomness = True

# ********** Run info END **********


def main():
    pipeline()
    #plot_elo()
    #comparison_trained()
    #comparison_from_scratch()
    #test_play()


def pipeline():
    # Train.
    self_play_and_store_versions(game_class, players, search_time, training_timer)
    # Tournament.
    start_elo_tournament(game_class, players, num_versions, num_elo_matches, match_randomness)


def plot_elo():
    from Misc.Plotter import plot_elo_ratings
    plot_elo_ratings(ConnectFour, num_versions)



# TODO: Remove everything below later. (Used for testing).


def test_play():
    play_player(
        game_class=ConnectFour,
        player=NnAlphaBetaPlayer(),
        search_time=None,
        version=20
    )


def train_and_store():
    players = [NnAlphaBetaPlayer(), RandomPlayer()]
    self_play_and_store_versions(
        TicTacToe,
        players,
        search_time,
        training_timer
    )


def comparison_from_scratch():
    from Matchmaking.Comparison1v1 import compare_ex_it_from_scratch
    # Run Comparison with several iteration of self-play.
    players = [NnAlphaBetaPlayer(), RandomPlayer()]

    compare_ex_it_from_scratch(
        game_class=ConnectFour,
        players=players,
        search_time=search_time,
        num_matches=100,
        training_timer=training_timer,
        randomness=True # <---------------------------------- Remember!
    )


def comparison_trained():
    from Matchmaking.Comparison1v1 import compare_ex_it_trained
    players = [NnAlphaBetaPlayer(), StaticMinimaxPlayer()]
    versions = range(20)
    compare_ex_it_trained(
        game_class=ConnectFour,
        raw_players=players,
        num_matches=100,
        randomness=True,  # <---------------------------------- Remember!
        versions=versions
    )


def elo_tournament():
    players = [NnAlphaBetaPlayer(), RandomPlayer()]
    start_elo_tournament(
        game_class=TicTacToe,
        raw_players=players,
        num_versions=num_versions,
        num_matches=num_elo_matches,
        randomness=True # <---------------------------------- Remember!
    )


def normal_exit_test():
    from Misc.Debugger import debug_display_win_moves
    # Run One iteration of self.play
    player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    if player.ex_it_algorithm.growing_search is None:
        player.set_search_time(search_time)
    player.start_ex_it(training_timer)

    # Display the agents calculations in predefined scenarios.
    debug_display_win_moves(player)


if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()')
