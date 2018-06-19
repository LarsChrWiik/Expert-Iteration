
import matplotlib.pyplot as plt
import csv


def plot_result(folder, filename):
    base_path = './Statistics' + "/" + folder
    path = base_path + "/" + filename + ".csv"
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file)
        data = [[float(row["win"]), float(row["loss"]), float(row["draw"])] for row in reader]

        print(data)
        iterations = __get_iterations(base_path=base_path)
        wins, losses, draws = __get_statistics(data=data, iterations=iterations)
        plt.plot(wins, label="Wins")
        plt.plot(losses, label="Losses")
        plt.plot(draws, label="Draws")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)
        plt.axis(ymin=0, ymax=1.0)
        plt.show()


def __get_iterations(base_path):
    path = base_path + "/" + "metadata" + ".txt"
    with open(path) as txt_file:
        content = txt_file.readlines()
        return float(content[2].split(" ")[-1])


def __get_statistics(data, iterations):
    wins = []
    losses = []
    draws = []
    for v in data:
        wins.append(v[0] / iterations)
        losses.append(v[1] / iterations)
        draws.append(v[2] / iterations)
    return wins, losses, draws
