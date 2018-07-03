
from ExIt.ExpertIteration import ExpertIteration
from keras.models import load_model
import os.path
from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange


model_folder = "Trained_Models"
meta_file = "meta"
model_file_type = ".h5"
txt_file_type = ".txt"


def self_play_and_store_versions(ex_it_algorithm: ExpertIteration,
                                 search_time, iterations, epochs, game_class):
    """ This process will produce several versions of an
        ExIt algorithm during self-play training """
    ex_it_algorithm.set_game(game_class)

    path = init_self_play_folders(
        ex_it_algorithm=ex_it_algorithm,
        search_time=search_time,
        iterations=iterations,
        epochs=epochs,
        game_class=game_class
    )

    print("Training: " + ex_it_algorithm.__name__)

    # Start self-play iterations.
    with trange(iterations) as t:
        for i in t:
            # Self-train.
            ex_it_algorithm.start_ex_it(
                epochs=epochs,
                search_time=search_time,
                verbose=False
            )
            # Save model.
            save(
                model=ex_it_algorithm.apprentice.model,
                path=path,
                iteration=str(i+1)
            )


def save(model, path, iteration):
    # Creates a HDF5 file.
    model.save(path + "/" + iteration + model_file_type)


def load(game_name, algorithm_name, iteration):
    path = get_model_folder_path() + "/" + game_name + "/" + algorithm_name \
           + "/" + iteration + model_file_type
    return load_model(path)


def init_self_play_folders(ex_it_algorithm, search_time, iterations, epochs, game_class):
    game_name = game_class.__name__
    algorithm_name = ex_it_algorithm.__name__

    create_base_folder_if_needed()
    create_game_folder_if_needed(game_name)
    create_algorithm_folder(game_name, algorithm_name)

    path = get_model_folder_path() + "/" + game_name + "/" + algorithm_name
    save_meta_information(
        path=path,
        search_time=search_time,
        iterations=iterations,
        epochs=epochs,
        algorithm_name=algorithm_name,
        ex_it_algorithm=ex_it_algorithm
    )
    return path


def save_meta_information(path, search_time, iterations, epochs, algorithm_name, ex_it_algorithm):
    with open(path + "/" + meta_file + txt_file_type, 'x') as file:
        file.write("algorithm_name = " + str(algorithm_name) + "\n")
        file.write("time = " + str(datetime.now().strftime('%Y-%m-%d___%H:%M:%S')) + "\n")
        file.write("search_time = " + str(search_time) + "\n")
        file.write("iterations = " + str(iterations) + "\n")
        file.write("epochs = " + str(epochs) + "\n")
        file.write("\n")
        file.write("optimizer = " + str(type(ex_it_algorithm.apprentice.optimizer).__name__) + "\n")
        file.write("n_neurons = " + str(ex_it_algorithm.apprentice.n_neurons) + "\n")
        file.write("n_layers = " + str(ex_it_algorithm.apprentice.n_layers) + "\n")
        file.write("dropout_rate = " + str(ex_it_algorithm.apprentice.dropout_rate) + "\n")


def create_algorithm_folder(game_name, algorithm_name):
    path = get_model_folder_path() + "/" + game_name + "/" + algorithm_name
    if Path(path).exists():
        raise Exception("Game name: '" + algorithm_name + "' already exists")
    else:
        os.makedirs(path)


def create_game_folder_if_needed(game_name):
    path = get_model_folder_path() + "/" + game_name
    if not Path(path).exists():
        os.makedirs(path)


def create_base_folder_if_needed():
    path = get_model_folder_path()
    if not Path(path).exists():
        os.makedirs(path)


def get_model_folder_path():
    return "./" + model_folder
