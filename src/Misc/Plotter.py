
from Misc.DiskHandler import read_ratings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def add_margins(ax, scale):
    left = ax.get_xlim()[0]
    right = ax.get_xlim()[1]
    bottom = ax.get_ylim()[0]
    top = ax.get_ylim()[1]

    x = abs(left - right)
    x_difference = x*scale
    y = abs(bottom - top)
    y_difference = y*scale

    ax.set_xlim(left=left-x_difference, right=right+x_difference)
    ax.set_ylim(bottom=bottom-y_difference, top=top+y_difference)


def plot_elo_ratings(game_class, num_versions):
    tournament = read_ratings(game_class, num_versions)
    colors = sns.color_palette("Set1", n_colors=len(tournament), desat=.5)

    ax = None
    for i, dic in enumerate(tournament):
        player_key = list(dic.keys())[0]
        player_info = dic[player_key]
        x = np.array(player_info["elo"])
        uncertainty = np.array([player_info["uncertainty+"], player_info["uncertainty-"]])
        data = x + uncertainty
        if player_key.startswith("ExIt"):
            # ExIt players.
            ax = sns.tsplot(data=data, time=range(1, num_versions+1), color=colors[i],
                            condition=player_key, marker="o", linestyle="-",
                            ax=ax)
        else:
            # Non-ExIt players.
            data = [[a, a] for a in data.tolist()]
            ax = sns.tsplot(data=data, time=[-100, num_versions*2], color=colors[i], linewidth=2,
                       condition=player_key, marker="", linestyle="--", ax=ax)

    plt.axvline(x=0, color="grey", linestyle="--")

    ax.set_xlim(0, num_versions)

    # Legend above the diagram.
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc=len(tournament),
        ncol=1,
        mode="expand"
    )

    ax.set(ylabel='Elo Rating', xlabel='Apprentice version')
    plt.subplots_adjust(top=.75, bottom=0.15)
    add_margins(ax, scale=0.1)

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
