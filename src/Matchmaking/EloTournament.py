
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from Players.BasePlayers import set_indexes
from Matchmaking.GameHandler import GameHandler
from Training import load
from pathlib import Path
import os.path


def start_tournament(game_class, players: [BasePlayer], trained_iterations,
                     num_matches=100, randomness=True):
    # TODO: This might not be used.
    set_indexes(players)

    # Generate all permutations of matches (list of 2-tuples of player index).
    matches_index = []
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if p1 is not p2:
                matches_index.append((i, j))

    # Create folders.
    create_path_folders_if_needed("Elo", str(game_class.__name__))
    base_path = "./Elo/" + str(game_class.__name__)
    # Ensure no overwriting is taking place.
    if Path(base_path + "/1.txt").exists():
        raise Exception("'" + base_path + "/1.txt' already exist. ")

    # Start tournament iteration.
    for i in range(trained_iterations):
        load_trained_model(game_class, players, i+1)

        # Match players.
        for j in range(num_matches):
            for i1, i2 in matches_index:
                game_handler = GameHandler(game_class, [players[i1], players[i2]], randomness)
                game_handler.play_game_until_finish()

                with open(base_path + "/" + str(i+1) + ".pgn", 'a') as file:
                    file.write("[Game \"" + str(game_class.__name__) + "\"]\n")
                    file.write("[White \"" + str(players[i1].__name__()) + "\"]\n")
                    file.write("[Black \"" + str(players[i2].__name__()) + "\"]\n")
                    file.write("[Result \"" + game_handler.result_text + "\"]\n")
                    file.write("\n")
                    file.write(game_handler.move_text)
                    file.write("\n")
                    file.write("\n")
                    file.write("\n")


def load_trained_model(game_class, players, i):
    for p in players:
        if isinstance(p, BaseExItPlayer):
            trained_model = load(
                game_name=game_class.__name__,
                algorithm_name=p.__name__(),
                iteration=str(i)
            )
            p.ex_it_algorithm.apprentice.set_model(trained_model)


def create_path_folders_if_needed(*args):
    """ Created folders for all arguments in args """
    path = "./"
    for arg in args:
        path += arg
        if not Path(path).exists():
            os.makedirs(path)
        path += "/"
