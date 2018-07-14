
from Misc.DiskHandler import read_ratings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def plot_elo_ratings(game_class, versions):
    tournament = read_ratings(game_class, versions)
    colors = sns.color_palette("Set1", n_colors=len(tournament), desat=.5)

    for i, dic in enumerate(tournament):
        player_key = list(dic.keys())[0]
        player_info = dic[player_key]
        x = np.array([0] + player_info["elo"])
        uncertainty = np.array([[0] + player_info["uncertainty+"], [0] + player_info["uncertainty-"]])
        data = x + uncertainty
        ax = sns.tsplot(data=data, color=colors[i], legend=True, condition=player_key, marker="o")

    # Legend above the diagram.
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc=len(tournament),
        ncol=1,
        mode="expand",
        borderaxespad=0.
    )

    ax.set(ylabel='Elo Rating', xlabel='Apprentice version')
    plt.subplots_adjust(top=.75, bottom=0.15)
    plt.show()


# TODO: Re-implement.
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
