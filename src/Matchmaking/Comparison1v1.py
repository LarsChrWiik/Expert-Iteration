
from Players.Players import BasePlayer, BaseExItPlayer
from Games.GameLogic import GameResult
from Matchmaking.GameHandler import GameHandler
from operator import add
from datetime import datetime
import os


def match(game_class, players, randomness):
    """ Starts a new game match between players and returns the result.
        Return a list of GameResult for each player according to Game Index. """
    game_handler = GameHandler(game_class, players, randomness)
    game_handler.play_game_until_finish()
    return game_handler.get_result()


class Comparison1v1:
    """ Logic for matching players """

    def __init__(self, game_class, players: [BasePlayer]):
        self.game_class = game_class
        self.players = players
        self.statistics = None

        # Assign the players a unique 'player index' (this index is constant).
        for i, p in enumerate(players):
            p.index = i

    def compare_ex_it(self, num_train_epoch, search_time,
                      num_matches, num_iteration, randomness):
        """ Compare players through several iterations.
            Self play is enabled for ExIt players between iterations.
            NB: This process never ends if num_iteration = None. """

        self.statistics = Statistics(
            num_train_epoch=num_train_epoch,
            search_time=search_time,
            num_matches=num_matches,
            num_iteration=num_iteration,
            players=self.players
        )

        # Let the players know which game they are playing.
        for p in self.players:
            if isinstance(p, BaseExItPlayer):
                if p.ex_it_algorithm.apprentice.model is None:
                    p.set_game(self.game_class)

        """ Compare players through several iterations of self-play.
            This process accepts non-ExIt player as well such as RandomPlayer. """
        i = 0
        while True:
            results = self.__compare(num_matches=num_matches, randomness=randomness)
            self.statistics.save(results=results)
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
                    epochs=num_train_epoch,
                    search_time=search_time
                )

    def __compare(self, num_matches, randomness):
        """ Compare players and store the statistics """
        # 2D list: [win, lose, draw]. Position = player index. (NOT GAME INDEX).
        results = [GameResult.get_new_result_list() for _ in self.players]

        for _ in range(num_matches):
            # Play game between players.
            game_result = match(self.game_class, self.players, randomness)
            # Get list of results for each player according to player index.
            game_result_list = GameResult.get_players_result_list_(game_result)
            # Add the result to the list of total results.
            for i, result in enumerate(results):
                results[i] = list(map(add, result, game_result_list[i]))
            self.__rearrange_players()
        return results

    def __rearrange_players(self):
        """ Moves the first player to the last position """
        player = self.players.pop(0)
        self.players.append(player)


class Statistics:
    """ Class containing logic for storing statistics to disk """

    metadata = "metadata"
    folder = "./Statistics/"

    def __init__(self, num_train_epoch, search_time,
                 num_matches, num_iteration, players):
        folder_name = self.make_new_folder()
        self.base_path = Statistics.folder + folder_name + "/"

        # Write initial information
        with open(self.__get_path(file_name=Statistics.metadata, file_type="txt"), 'w') as file:
            file.write("num_train_epoch = " + str(num_train_epoch) + "\n")
            file.write("search_time = " + str(search_time) + "\n")
            file.write("num_matches = " + str(num_matches) + "\n")
            file.write("num_iteration = " + str(num_iteration) + "\n")
            file.write("\n")
            file.write("Players:\n")
            for p in players:
                file.write("- " + str(type(p).__name__) + " id = " + str(p.index) + "\n")

        for p in players:
            with open(self.__get_path(file_name=str(p.index)), 'w') as file:
                file.write("win,loss,draw" + "\n")

    def save(self, results):
        """ Save statistic about the game """
        print(results)
        for i, result in enumerate(results):
            with open(self.__get_path(file_name=str(i)), 'a') as file:
                file.write(self.__convert_result(result=result) + "\n")

    def __get_path(self, file_name, file_type="csv"):
        return self.base_path + file_name + str(".") + file_type

    @staticmethod
    def make_new_folder():
        folder_name = str(datetime.now().strftime('%Y-%m-%d___%H-%M-%S'))
        if not os.path.exists(folder_name):
            os.makedirs(Statistics.folder + folder_name)
        else:
            raise Exception('Folder name already exists!')
        return folder_name

    @staticmethod
    def __convert_result(result):
        result_new = ""
        for i, v in enumerate(result):
            result_new += str(v)
            if i != len(result) - 1:
                result_new += ","
        return result_new
