
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from Matchmaking.GameHandler import GameHandler
from Misc.DiskHandler import create_elo_folders, save_game_to_pgn, \
    load_trained_model, create_elo_meta_file
from tqdm import trange


def start_elo_tournament(game_class, players_classes, num_versions, num_matches=250, randomness=True):
    """ Match players in a tournament and writes matches to PGN files """

    base_path = create_elo_folders(game_class)
    create_elo_meta_file(base_path, game_class, players_classes, num_matches, randomness)

    players = load_trained_model(game_class, players_classes, num_versions)
    for p in players:
        print(p.__name__)
    match_permutations = get_all_match_permutations(players)
    # Match players with all permutations 'num_matches' times.
    with trange(num_matches) as t:
        for _ in t:
            for i1, i2 in match_permutations:
                p1, p2 = players[i1], players[i2]
                game_handler = GameHandler(game_class, [p1, p2], randomness)
                game_handler.play_game_until_finish()
                save_game_to_pgn(base_path, game_handler, p1, p2)


def get_all_match_permutations(players):
    """ Generate a list of 2-tuples of all match permutations between players.
        Total of len(players) * (len(players)-1) tuples. """
    match_permutations = []
    for i, p1 in enumerate(players):
        for j, p2 in enumerate(players):
            if p1 is not p2:
                match_permutations.append((i, j))
    return match_permutations
