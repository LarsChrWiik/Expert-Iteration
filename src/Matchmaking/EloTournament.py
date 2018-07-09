
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from Matchmaking.GameHandler import GameHandler
from Misc.DiskHandler import create_elo_folders, save_game_to_pgn, load_model


def start_elo_tournament(game_class, players: [BasePlayer], trained_iterations,
                         num_matches=100, randomness=True):
    """ Match players in a tournament and writes matches to PGN files """
    match_permutations = get_all_match_permutations(players)
    base_path = create_elo_folders(game_class)

    # Start tournament iteration.
    for i in range(trained_iterations):
        load_trained_model(game_class, players, i)

        # Match players with all permutations 'num_matches' times.
        for _ in range(num_matches):
            for i1, i2 in match_permutations:
                p1, p2 = players[i1], players[i2]
                game_handler = GameHandler(game_class, [p1, p2], randomness)
                game_handler.play_game_until_finish()
                save_game_to_pgn(base_path, game_handler, p1, p2, i)


def get_all_match_permutations(players):
    """ Generate a list of 2-tuples of all match permutations between players """
    match_permutations = []
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if p1 is not p2:
                match_permutations.append((i, j))
    return match_permutations


def load_trained_model(game_class, players, i):
    """ Load trained model into the players """
    for p in players:
        if isinstance(p, BaseExItPlayer):
            trained_model = load_model(
                game_name=game_class.__name__,
                algorithm_name=p.__name__(),
                iteration=str(i+1)
            )
            p.ex_it_algorithm.apprentice.set_model(trained_model)
