
from Misc.DiskHandler import create_training_folders, create_training_meta_file, save_model
from Players.BasePlayers import BaseExItPlayer


def self_play_and_store_versions(game_class, players, search_time, training_timer):
    """ This process will produce several versions of an
        ExIt algorithm during self-play training """

    player_and_path = []
    for p in players:
        if not isinstance(p, BaseExItPlayer):
            continue
        p.set_game(game_class)
        if not p.ex_it_algorithm.use_growing_search_time:
            p.set_search_time(search_time)
        base_path = create_training_folders(game_class, p)
        create_training_meta_file(base_path, p, search_time, training_timer)
        player_and_path.append((p, base_path))

    for i in range(training_timer.num_versions):
        for p, path in player_and_path:
            # Self-train.
            p.start_ex_it(training_timer, verbose=True)
            # Save model.
            save_model(p.ex_it_algorithm.apprentice.model, path, version=str(i + 1))
