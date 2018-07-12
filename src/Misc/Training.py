
from ExIt.ExpertIteration import ExpertIteration
from Misc.DiskHandler import create_training_folders, create_training_meta_file, save_model


def self_play_and_store_versions(game_class, ex_it_algorithm: ExpertIteration,
                                 search_time, training_timer):
    """ This process will produce several versions of an
        ExIt algorithm during self-play training """
    ex_it_algorithm.set_game(game_class)

    # Create folders and meta file and get get base path.
    base_path = create_training_folders(game_class, ex_it_algorithm)
    create_training_meta_file(base_path, ex_it_algorithm, search_time, training_timer)

    print("Training: " + ex_it_algorithm.__name__)

    # Start self-play iterations.
    for i in range(training_timer.num_versions):
        # Self-train.
        ex_it_algorithm.start_ex_it(
            training_timer=training_timer,
            search_time=search_time,
            verbose=True
        )
        # Save model.
        save_model(
            model=ex_it_algorithm.apprentice.model,
            base_path=base_path,
            iteration=str(i+1)
        )
