
from Games.TicTacToe import TicTacToe
from Games.ConnectFour import ConnectFour
from Games.ConnectSix import ConnectSix
from Games.Othello import Othello
from Players.Players import *
from Matchmaking.EloTournament import start_elo_tournament
from Matchmaking.Comparison1v1 import compare_ex_it_from_scratch
from Misc.Training import self_play_and_store_versions
from Misc.TrainingTimer import get_seconds
from Misc.TrainingTimer import TrainingTimer
from ExIt.Policy import Policy
from Misc.PlayGameCLI import play, play_player
import numpy as np
np.set_printoptions(suppress=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Game.
game_class = Othello()

# Players to compare.
players = [
    NnMinimaxPlayer(use_ab=True),
    #NnMctsPlayer(growing_search_time=True),
    RandomPlayer(),
    BruteForcePlayer(depth=1),
    BruteForcePlayer(depth=2)
]
# Search time for each player.
search_time = get_seconds(s=0.1)

# Total time for each player to self-train.
time_limit = get_seconds(d=1)
# Number of versions to be trained.
num_versions = 24
# Timer. NB: Each version is trained for time_limit / num_versions time).
training_timer = TrainingTimer(time_limit, num_versions)

# Number of matches to compare the players. This is used to calculate Elo.
# More matches = more certain of elo scores.
num_elo_matches = 10000
# Chance of random action when advancing.
match_randomness = 0.1


def main():
    # This will store trained versions of players in the Trained_models folder.
    self_play_and_store_versions(game_class, players, search_time, training_timer)

    #compare_ex_it_from_scratch(game_class, players, search_time, 100, training_timer, randomness=0.1)

    # This will generate a PGN file in Elo folder.
    start_elo_tournament(game_class, players, num_versions, num_elo_matches, match_randomness)
    # Plot elo scores from rating.txt file in Elo folder after using Bayesian Elo.
    #plot_elo(game_class, num_versions)
    # Play against a player (either static or ExIt player with specified version).
    #play_player(game_class, NnMinimaxPlayer(use_ab=True, growing_search_time=True), search_time=1.0, version=20)


def plot_elo(game_class, num_versions):
    from Misc.Plotter import plot_elo_ratings
    plot_elo_ratings(game_class, num_versions)


if __name__ == "__main__":
    main()
    #import cProfile
    #cProfile.run('main()')
