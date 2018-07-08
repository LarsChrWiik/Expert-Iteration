
from datetime import datetime
import os


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
