
from pathlib import Path
from datetime import datetime
from Players.BasePlayers import BaseExItPlayer
from keras.models import load_model as load_keras_model
from Games.GameLogic import GameResult
import os.path

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


def load_trained_model(game_class, players, i):
    """ Load trained model into the players """
    for p in players:
        if isinstance(p, BaseExItPlayer):
            trained_model = load_model(
                game_name=game_class.__name__,
                ex_it_algorithm=p.ex_it_algorithm,
                iteration=str(i)
            )
            p.ex_it_algorithm.apprentice.set_model(trained_model)


# ******************** ELO ********************


def create_elo_folders(game_class):
    """ Create Elo folders and return the base path.
        Also ensures that no overwriting is taking place. """
    base_path = create_path_folders_if_needed("Elo", game_class.__name__)
    if Path(base_path + "/1.pgn").exists():
        raise Exception("'" + base_path + "/1.pgn' already exist. ")
    return base_path


def save_game_to_pgn(base_path, game_handler, p1, p2, i):
    """ Convert game match into pgn format and writes it to file """
    with open(base_path + "/" + str(i+1) + ".pgn", 'a') as file:
        file.write("[Game \"" + str(game_handler.game_class.__name__) + "\"]\n")
        file.write("[White \"" + p1.__name__() + "\"]\n")
        file.write("[Black \"" + p2.__name__() + "\"]\n")
        file.write("[Result \"" + game_handler.result_text + "\"]\n")
        file.write("\n")
        file.write(game_handler.move_text)
        file.write("\n")
        file.write("\n")
        file.write("\n")


# ******************** TRAINING ********************


def save_model(model, base_path, iteration):
    """ Store the model as a HDF5 file """
    model.save(base_path + "/" + iteration + ".h5")


def load_model(game_name, ex_it_algorithm, iteration):
    return load_keras_model(
        "./Trained_models/" + game_name + "/" + ex_it_algorithm.__name__ + "/" + iteration + ".h5"
    )


def count_trained_models(game_class, players):
    files = os.listdir("./Trained_models/" + game_class.__name__ + "/" + players[0].ex_it_algorithm.__name__ + "/")
    return len([x for x in files if x[-3:] == ".h5"])


def create_training_folders(game_class, ex_it_algorithm):
    base_path = create_path_folders_if_needed("Trained_models", game_class.__name__, ex_it_algorithm.__name__)
    if Path(base_path + "/meta.txt").exists():
        raise Exception("'" + base_path + "/meta.txt' already exist. ")
    return base_path


def create_training_meta_file(base_path, ex_it_algorithm, search_time, training_timer):
    with open(base_path + "/meta.txt", 'x') as file:
        file.write("Datetime = " + str(datetime.now().strftime('%Y-%m-%d___%H:%M:%S')) + "\n")
        file.write("Training time = " + str(training_timer.time_limit) + "\n")
        file.write("Number of versions = " + str(training_timer.num_versions) + "\n")
        file.write("Time pr version = "
                   + str(training_timer.time_limit / training_timer.num_versions) + "\n")
        file.write("Search_time = " + str(search_time) + "\n")
        file.write("\n")
        file.write("Policy = " + ex_it_algorithm.policy_str + "\n")
        file.write("State branch degree = " + str(ex_it_algorithm.state_branch_degree) + "\n")
        file.write("Dataset type = " + type(ex_it_algorithm.data_set).__name__ + "\n")
        file.write("\n")
        file.write(ex_it_algorithm.__name__ + "\n")
        write_ex_it_model_info(file, ex_it_algorithm)


# ******************** COMPARISON 1v1 ********************


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
            file.write(p.__name__() + "\n")
            if isinstance(p, BaseExItPlayer):
                write_ex_it_model_info(file, p.ex_it_algorithm)
            if i != len(players) - 1:
                file.write("\n")


def create_comparison_files(base_path, players):
    for p in players:
        with open(base_path + "/" + p.__name__() + ".csv", 'x') as file:
            file.write("iteration,win,loss,draw" + "\n")


def save_comparison_result(base_path, results: [GameResult], i, p):
    """ Saves results to disk.
        NB: This function assumes that results_list is ordered by player index """
    with open(base_path + "/" + p.__name__() + ".csv", 'a') as file:
        file.write(str(i) + "," + GameResult.get_string_results(results) + "\n")
