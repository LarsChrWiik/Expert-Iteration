
from Misc.DiskHandler import read_ratings, get_comparison_base_path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def add_y_margins_inside(ax, num_versions, y_scale):
    bottom = ax.get_ylim()[0]
    top = ax.get_ylim()[1]

    y = abs(bottom - top)
    y_difference = y * y_scale

    ax.set_ylim(bottom=bottom-y_difference, top=top+y_difference)
    ax.set_xlim(0, num_versions + 1)


def plot_elo_ratings(game_class, num_versions):
    tournament = read_ratings(game_class)
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
            ax = sns.tsplot(data=data, time=[0, num_versions+1], color=colors[i], linewidth=2,
                            condition=player_key, marker="", linestyle="--", ax=ax)

    ax.set_xlim(0, num_versions)

    ax.set(ylabel='Elo Rating', xlabel='Apprentice version')
    plt.subplots_adjust(top=0.9 - len(tournament)*0.04, bottom=0.15)
    add_y_margins_inside(ax, num_versions, y_scale=0.1)

    # Legend above the diagram.
    ax.legend(
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc=3,
        borderaxespad=0.,
        mode="expand"
    )

    plt.show()


def plot_comparison1v1(folder, player):
    base_path = get_comparison_base_path(folder)
    path = base_path + player.__name__ + ".csv"
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file)
        data = [[float(row["win"]), float(row["loss"]), float(row["draw"])] for row in reader]

        wins = [row[0] for row in data]
        losses = [row[1] for row in data]
        draws = [row[2] for row in data]

        plt.plot(wins, label="Wins")
        plt.plot(losses, label="Losses")
        plt.plot(draws, label="Draws")
        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            loc=3,
            ncol=3,
            mode="expand",
            borderaxespad=0.
        )
        plt.ylim(0, wins[0] + losses[0] + draws[0])
        plt.show()
