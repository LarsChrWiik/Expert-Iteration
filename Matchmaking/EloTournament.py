
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from Matchmaking.GameHandler import GameHandler
from Misc.DiskHandler import create_elo_folders, save_game_to_pgn, \
    load_trained_models, create_elo_meta_file
from tqdm import tqdm, trange
import random


def start_elo_tournament(game_class, raw_players, num_versions, num_matches, randomness=True):
    """ Match players in a tournament and writes matches to PGN files """

    base_path = create_elo_folders(game_class)

    players = load_trained_models(game_class, raw_players, range(num_versions))
    player_indexes = range(len(players))

    create_elo_meta_file(base_path, game_class, raw_players, num_matches, num_versions, randomness)

    # Match players with all permutations 'num_matches' times.
    progress_bar = tqdm(range(num_matches))
    progress_bar.set_description("Tournament")
    matches_played = 0
    while matches_played < num_matches:
        for i1, p in enumerate(players):
            # Get index of any other player.
            i2 = i1
            while i2 == i1:
                i2 = random.choice(player_indexes)

            # Match the players.
            p1, p2 = players[i1], players[i2]
            game_handler = GameHandler(game_class, [p1, p2], randomness)
            game_handler.play_game_until_finish()
            save_game_to_pgn(base_path, game_handler, p1, p2)
            progress_bar.update(1)
            matches_played += 1
            if matches_played >= num_matches:
                break

    progress_bar.close()
