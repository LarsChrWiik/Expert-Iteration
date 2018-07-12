
from Players.Players import BaseExItPlayer
from Players.BasePlayers import set_indexes
from Games.GameLogic import GameResult
from Matchmaking.GameHandler import match
from Misc.DiskHandler import create_comparison_folders, create_comparison_meta_file, \
    create_comparison_files, save_comparison_result, load_trained_model
from operator import add


def compare_ex_it_trained(game_class, players, num_matches, randomness, version):
    """ Compare trained players """
    set_indexes(players)

    load_trained_model(game_class, players, version)

    # Create necessary folders and files and get base path.
    base_path = create_comparison_folders()
    create_comparison_meta_file(base_path, players, num_matches, None, None, version)
    create_comparison_files(base_path, players)

    results_list = start_matches(game_class, players, num_matches, randomness)
    for p_index, results in enumerate(results_list):
        save_comparison_result(base_path, results, 1, [p for p in players if p.index == p_index][0])
    print(results_list)


def compare_ex_it_from_scratch(game_class, players, search_time,
                               num_matches, training_timer, randomness):
    """ Compare players through several iterations.
        Self play is enabled for ExIt players between iterations. """

    set_indexes(players)

    # Create necessary folders and files and get base path.
    base_path = create_comparison_folders()
    create_comparison_meta_file(base_path, players, num_matches, training_timer, search_time)
    create_comparison_files(base_path, players)

    # Let the players know which game they are playing.
    for p in players:
        if isinstance(p, BaseExItPlayer):
            if p.ex_it_algorithm.apprentice.model is None:
                p.set_game(game_class)

    """ Compare players through several iterations of self-play.
        This process accepts non-ExIt player as well such as RandomPlayer. """
    for i in range(training_timer.num_versions):
        # Let BaseExItPlayers train.
        for player in players:
            if isinstance(player, BaseExItPlayer):
                player.start_ex_it(
                    training_timer=training_timer,
                    search_time=search_time
                )
        # Start match and get results.
        results_list = start_matches(game_class, players, num_matches, randomness)
        for p_index, results in enumerate(results_list):
            save_comparison_result(base_path, results, i+1, [p for p in players if p.index == p_index][0])
        rearrange_players(players)
        print(results_list)


def start_matches(game_class, players, num_matches, randomness):
    """ Play several games between the players """
    # 2D list: [win, lose, draw]. Position = player index.
    results = [GameResult.get_new_result_list() for _ in players]

    for _ in range(num_matches):
        game_result = match(game_class, players, randomness)
        game_result_list = GameResult.get_players_result_list_(game_result)
        for i, result in enumerate(results):
            results[i] = list(map(add, result, game_result_list[i]))
        rearrange_players(players)
    return results


def rearrange_players(players):
    """ Moves the first player to the last position """
    player = players.pop(0)
    players.append(player)
