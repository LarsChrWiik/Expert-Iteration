
from pathlib import Path
from datetime import datetime
from Players.BasePlayers import BaseExItPlayer
from keras.models import load_model as load_keras_model
from Games.GameLogic import GameResult
from tqdm import tqdm, trange
import os.path
import re
from copy import deepcopy


# ******************** GENERAL ********************


def create_path_folders_if_needed(*args):
    """ Created folders for all arguments in args if they does not exist """
    path = "./"
    for i, arg in enumerate(args):
        path += arg
        if not Path(path).exists():
            os.makedirs(path)
        if i != len(args)-1:
            path += "/"
    return path


def write_ex_it_model_info(file, ex_it_algorithm):
    file.write("   optimizer = " + type(ex_it_algorithm.apprentice.optimizer).__name__ + "\n")
    file.write("   n_neurons = " + str(ex_it_algorithm.apprentice.n_neurons) + "\n")
    file.write("   n_layers = " + str(ex_it_algorithm.apprentice.n_layers) + "\n")
    file.write("   dropout_rate = " + str(ex_it_algorithm.apprentice.dropout_rate) + "\n")


def load_trained_models(game_class, raw_players, versions):
    """ Load trained model into the players """
    players = []
    num_ex_it_players = len([1 for p in raw_players if isinstance(p, BaseExItPlayer)])
    progress_bar = tqdm(range(len(versions) * num_ex_it_players))
    progress_bar.set_description("load_trained_model")
    for p in raw_players:
        if isinstance(p, BaseExItPlayer):
            for v in versions:
                version = v+1
                new_player = p.new_player()
                new_player.__name__ = new_player.__name__ + "_model" + str(version)
                trained_model = load_model(
                    game_name=game_class.__name__,
                    ex_it_algorithm=new_player.ex_it_algorithm,
                    iteration=str(version)
                )
                new_player.ex_it_algorithm.apprentice.set_model(trained_model)
                players.append(new_player)
                progress_bar.update(1)
        else:
            players.append(p.new_player())
    progress_bar.close()
    return players


# ******************** ELO ********************


def create_elo_folders(game_class):
    """ Create Elo folders and return the base path.
        Also ensures that no overwriting is taking place. """
    base_path = create_path_folders_if_needed("Elo", game_class.__name__)
    if Path(base_path + "/tournament.pgn").exists():
        raise Exception("'" + base_path + "/tournament.pgn' already exist. ")
    return base_path


def create_elo_meta_file(base_path, game_class, raw_players, num_matches, num_versions, randomness):
    with open(base_path + "/meta.txt", 'x') as file:
        file.write("Datetime = " + str(datetime.now().strftime('%Y-%m-%d___%H:%M:%S')) + "\n")
        file.write("Game = " + game_class.__name__ + "\n")
        file.write("Randomness = " + str(randomness) + "\n")
        file.write("Number of versions = " + str(num_versions) + "\n")
        file.write("Number of matches = " + str(num_matches) + "\n")
        file.write("\n")
        file.write("Players: \n")
        for i, p in enumerate(raw_players):
            file.write("   - " + p.__name__ + "\n")


def save_game_to_pgn(base_path, game_handler, p1, p2):
    """ Convert game match into pgn format and writes it to file """
    with open(base_path + "/tournament.pgn", 'a') as file:
        file.write("[Game \"" + str(game_handler.game_class.__name__) + "\"]\n")
        file.write("[White \"" + p1.__name__ + "\"]\n")
        file.write("[Black \"" + p2.__name__ + "\"]\n")
        file.write("[Result \"" + game_handler.result_text + "\"]\n")
        file.write("\n")
        file.write(game_handler.move_text)
        file.write("\n")
        file.write("\n")
        file.write("\n")


def read_ratings(game_class):
    with open("./Elo/" + game_class.__name__ + "/ratings.txt", 'r') as file:
        tournament = {}
        lines = []
        for i, line in enumerate(file):
            if i == 0:
                continue
            words = line.split()
            if words[1].startswith("ExIt"):
                version = re.findall(r'\d+', str(words[1]))[-1]
                version = int(version)
            else:
                version = 1
            lines.append((version, words))

        # Make sure versions are added in order.
        lines.sort()

        for version, words in lines:
            def add_info(player_name, info):
                if player_name in tournament.keys():
                    for key, value in tournament[player_name].items():
                        tournament[player_name][key].append(info[key][0])
                else:
                    tournament[player_name] = deepcopy(info)

            info = {
                "elo": [float(words[2])],
                "uncertainty-": [float(words[3])],
                "uncertainty+": [-float(words[4])],
                "games": [int(words[5])],
                "score": [float(words[6][:-1])],
                "oppo": [float(words[7])],
                "draws": [float(words[8][:-1])]
            }

            if words[1].startswith("ExIt"):
                player_name = str(words[1][:-(6 + len(str(version)))])
                add_info(player_name, info)
            else:
                player_name = words[1]
                add_info(player_name, info)

        # Convert to list and sort:
        tournament = [(max(value["elo"]), {key: value}) for key, value in tournament.items()]
        tournament.sort(reverse=True)
        tournament = [dic for elo, dic in tournament]

        return tournament


# ******************** TRAINING ********************


def save_model(model, base_path, iteration):
    """ Store the model as a HDF5 file """
    model.save(base_path + "/" + iteration + ".h5")


def load_model(game_name, ex_it_algorithm, iteration):
    return load_keras_model(
        "./Trained_models/" + game_name + "/" + ex_it_algorithm.__name__ + "/" + iteration + ".h5"
    )


def create_training_folders(game_class, p):
    base_path = create_path_folders_if_needed("Trained_models", game_class.__name__, p.__name__)
    if Path(base_path + "/meta.txt").exists():
        raise Exception("'" + base_path + "/meta.txt' already exist. ")
    return base_path


def create_training_meta_file(base_path, p: BaseExItPlayer, search_time, training_timer):
    with open(base_path + "/meta.txt", 'x') as file:
        file.write("Datetime = " + str(datetime.now().strftime('%Y-%m-%d___%H:%M:%S')) + "\n")
        file.write("Training time = " + str(training_timer.time_limit) + "\n")
        file.write("Number of versions = " + str(training_timer.num_versions) + "\n")
        file.write("Training time pr version = "
                   + str(training_timer.time_limit / training_timer.num_versions) + "\n")
        file.write("Search_time = " + str(search_time) + "\n")
        file.write("\n")
        file.write("Policy = " + str(p.ex_it_algorithm.policy.value) + "\n")
        file.write("State branch degree = " + str(p.ex_it_algorithm.state_branch_degree) + "\n")
        file.write("Dataset type = " + type(p.ex_it_algorithm.memory).__name__ + "\n")
        file.write("\n")
        file.write(p.__name__ + "\n")
        write_ex_it_model_info(file, p.ex_it_algorithm)


# ******************** COMPARISON 1v1 ********************


def get_comparison_base_path(folder):
    return './Comparison1v1' + "/" + folder + "/"


def create_comparison_folders():
    return create_path_folders_if_needed(
        "Comparison1v1", str(datetime.now().strftime('%Y-%m-%d___%H-%M-%S'))
    )


def create_comparison_meta_file(base_path, players, num_matches, training_timer, search_time, version=None):
    with open(base_path + "/meta.txt", 'x') as file:
        file.write("Date time = " + str(datetime.now().strftime('%Y-%m-%d___%H:%M:%S')) + "\n")
        file.write("Number of matches = " + str(num_matches) + "\n")
        if (training_timer, search_time) != (None, None):
            file.write("Training time = " + str(training_timer.time_limit) + "\n")
            file.write("Number of versions = " + str(training_timer.num_versions) + "\n")
            file.write("Time pr version = "
                       + str(training_timer.time_limit / training_timer.num_versions) + "\n")
            file.write("Search time = " + str(search_time) + "\n")
        if version is not None:
            file.write("version = " + str(version) + "\n")
        file.write("\n")
        for i, p in enumerate(players):
            file.write(p.__name__ + "\n")
            if isinstance(p, BaseExItPlayer):
                write_ex_it_model_info(file, p.ex_it_algorithm)
            if i != len(players) - 1:
                file.write("\n")


def create_comparison_files(base_path, players):
    for p in players:
        with open(base_path + "/" + p.__name__ + ".csv", 'x') as file:
            file.write("iteration,win,loss,draw" + "\n")


def save_comparison_result(base_path, results: [GameResult], i, p):
    """ Saves results to disk.
        NB: This function assumes that results_list is ordered by player index """
    with open(base_path + "/" + p.__name__ + ".csv", 'a') as file:
        file.write(str(i) + "," + GameResult.get_string_results(results) + "\n")
