
from Misc.DiskHandler import create_training_folders, create_training_meta_file, save_model
from Players.BasePlayers import BaseExItPlayer


def self_play_and_store_versions(game_class, players, search_time, training_timer):
    """ This process will produce several versions of an
        ExIt algorithm during self-play training """

    for p in players:
        if not isinstance(p, BaseExItPlayer):
            continue
        p.set_game(game_class)
        # Create folders and meta file and get get base path.
        base_path = create_training_folders(game_class, p)
        create_training_meta_file(base_path, p, search_time, training_timer)

        # Start self-play iterations.
        for i in range(training_timer.num_versions):
            # Self-train.
            p.start_ex_it(
                training_timer=training_timer,
                search_time=search_time,
                verbose=True
            )
            # Save model.
            save_model(
                model=p.ex_it_algorithm.apprentice.model,
                base_path=base_path,
                iteration=str(i+1)
            )
