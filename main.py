

from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from Games.ConnectSix import ConnectSix
from Games.TicTacToe4x4 import TicTacToe4x4
from Games.Othello import Othello
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

Q[s][a] = Expected Q values when taking action a from state s. 
N[s]    = Number of times state s was visited. 
P[s][a] = Predicted P values when taking action a from state s.
V[s]    = Predicted v value of state s.
***********************************************
"""


# ********** Run info START **********

# Game.
game_class = TicTacToe

# Players to compare.
players = [
    NnMctsPlayer(),
    NnMinimaxPlayer(use_ab=True),
    RandomPlayer(),
    BruteForcePlayer(depth=1),
    BruteForcePlayer(depth=2)
]
# Search time for each player.
search_time = get_seconds(s=0.25)

# Total time for each player to self-train.
time_limit = get_seconds(m=5)
# Number of versions to be trained.
num_versions = 20
# Timer. NB: Each version is trained for time_limit / num_versions time).
training_timer = TrainingTimer(time_limit, num_versions)

# Number of matches to compare the players. This is used to calculate Elo.
# More matches = more certain of elo scores.
num_elo_matches = 10000
# Chance of random action when advancing.
match_randomness = 0.1

# ********** Run info END **********




def main():
    pipeline()
    #plot_elo()
    #comparison_trained()
    #comparison_from_scratch()
    #test_play()


def pipeline():
    # Train.
    #self_play_and_store_versions(game_class, players, search_time, training_timer)
    # Tournament.
    start_elo_tournament(game_class, players, num_versions, num_elo_matches, match_randomness)


def plot_elo():
    from Misc.Plotter import plot_elo_ratings
    plot_elo_ratings(game_class, num_versions)



# TODO: Remove everything below later. (Used for testing).


def test_play():
    play_player(
        game_class=game_class,
        player=BruteForcePlayer(depth=2),
        search_time=None,
        version=None
    )


def train_and_store():
    players = [NnMinimaxPlayer(use_ab=True), RandomPlayer()]
    self_play_and_store_versions(
        TicTacToe,
        players,
        search_time,
        training_timer
    )


def comparison_from_scratch():
    from Matchmaking.Comparison1v1 import compare_ex_it_from_scratch
    # Run Comparison with several iteration of self-play.
    players = [NnMinimaxPlayer(use_ab=True), RandomPlayer()]

    compare_ex_it_from_scratch(
        game_class=TicTacToe,
        players=players,
        search_time=search_time,
        num_matches=100,
        training_timer=training_timer,
        randomness=match_randomness # <---------------------------------- Remember!
    )


def comparison_trained():
    from Matchmaking.Comparison1v1 import compare_ex_it_trained
    players = [NnMinimaxPlayer(use_ab=True), BruteForcePlayer(depth=2)]
    versions = [23]#range(20)
    compare_ex_it_trained(
        game_class=game_class,
        raw_players=players,
        num_matches=100,
        randomness=match_randomness,  # <---------------------------------- Remember!
        versions=versions
    )


if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()')
