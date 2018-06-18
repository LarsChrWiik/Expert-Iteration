
from Games.GameLogic import BaseGame
from Players.Players import BasePlayer, BaseExItPlayer
from Games.GameLogic import GameResult
from operator import add
from random import uniform
from Players.BasePlayer import assign_game_index
from Matchmaking.Statistics import Statistics


class GameHandler:
    """ Class to organize a game between players """

    def __init__(self, game: BaseGame, players: [BasePlayer]):
        self.game = game
        self.players = players
        assign_game_index(players=players)

    def start_game(self, rnd_prob=0.0):
        """ Starts the game and let the players play until the game has finished
            If the player is a ExIt player, then only the apprentice is used
            to decide the move. """
        while not self.game.is_game_over():
            if uniform(0, 1) < rnd_prob:
                self.players[self.game.turn].move_random(self.game)
            else:
                self.players[self.game.turn].move(self.game)

    def get_game_index_result(self):
        """ Return a list of GameResult for each player according to Game Index """
        result = []
        for p in self.players:
            result.append(self.game.get_result(player_index=p.game_index))
        return result


class Matchmaking:
    """ Logic for matching players """

    @staticmethod
    def match(game_class, players, rnd_prob=0.0):
        """ Starts a new game match between players and returns the result.
            Return a list of GameResult for each player according to Game Index. """
        game = game_class()
        game_handler = GameHandler(game=game, players=players)
        game_handler.start_game(rnd_prob=rnd_prob)
        game_index_result = game_handler.get_game_index_result()
        result = Matchmaking.get_index_result(
            game_index_result=game_index_result,
            players=players
        )
        return result

    @staticmethod
    def get_index_result(game_index_result, players):
        """ Convert rewards position from game player index to player index """
        result = [GameResult(None) for _ in range(len(game_index_result))]
        for p in players:
            result[p.index] = game_index_result[p.game_index]
        return result

    def __init__(self, game_class, players: [BasePlayer]):
        self.game_class = game_class
        self.players = players
        self.statistics = None

        # Assign the players a unique 'player index' (this index is constant).
        for index, p in enumerate(players):
            p.index = index

    def compare_ex_it(self, num_train_epoch, search_time,
                      num_matches, file_name, num_iteration):
        """ Compare players through several iterations.
            Self play is enabled for ExIt players between iterations.
            NB: This process never ends if num_iteration = None. """

        self.statistics = Statistics(
            file_name=file_name,
            num_train_epoch=num_train_epoch,
            search_time=search_time,
            num_matches=num_matches,
            num_iteration=num_iteration,
            players=self.players
        )

        # Let the players know which game they are playing.
        for p in self.players:
            if p.game_class is None:
                p.set_game(self.game_class)

        """ Compare players through several iterations of self-play.
            This process accepts non-ExIt player as well such as RandomPlayer. """
        i = 0
        while True:
            results = self.__compare(num_matches=num_matches)
            self.statistics.save(results=results, i=i)
            if num_iteration is not None and i >= num_iteration:
                break
            self.__train(num_train_epoch=num_train_epoch, search_time=search_time)
            i += 1

    def __train(self, num_train_epoch, search_time):
        """ Let the player train using ExIt self-play if the player
            is instance of BaseExItPlayer """
        for player in self.players:
            if isinstance(player, BaseExItPlayer):
                player.start_ex_it(
                    num_iteration=num_train_epoch,
                    randomness=True,
                    search_time=search_time
                )

    def __compare(self, num_matches):
        """ Compare players and store the statistics """
        # 2D list: [win, lose, draw]. Position = player index. (NOT GAME INDEX).
        results = [GameResult.get_new_result_list() for _ in self.players]

        for _ in range(num_matches):
            # Play game between players.
            game_result = Matchmaking.match(game_class=self.game_class, players=self.players)
            # Get list of results for each player.
            game_result = GameResult.get_players_result_list_(result=game_result)
            # Add the result to the list of total results.
            for i, result in enumerate(results):
                results[i] = list(map(add, result, game_result[i]))
            self.__rearrange_players()
        return results

    def __rearrange_players(self):
        """ Moves the first player to the last position """
        player = self.players.pop(0)
        self.players.append(player)
