
from ExIt.ExpertIteration import ExpertIteration
from tqdm import trange
from Misc.DiskHandler import create_training_folders, create_training_meta_file, save_model


def self_play_and_store_versions(ex_it_algorithm: ExpertIteration,
                                 search_time, iterations, epochs, game_class):
    """ This process will produce several versions of an
        ExIt algorithm during self-play training """
    ex_it_algorithm.set_game(game_class)

    # Create folders and meta file and get get base path.
    base_path = create_training_folders(game_class, ex_it_algorithm)
    create_training_meta_file(base_path, ex_it_algorithm, search_time, iterations, epochs)

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
            save_model(
                model=ex_it_algorithm.apprentice.model,
                base_path=base_path,
                iteration=str(i+1)
            )
