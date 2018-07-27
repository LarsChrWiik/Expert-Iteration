
from Misc.DiskHandler import read_ratings, get_comparison_base_path, load_elo_version_time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def add_y_margins_inside(ax, max_time, num_versions, y_scale):
    bottom = ax.get_ylim()[0]
    top = ax.get_ylim()[1]

    y = abs(bottom - top)
    y_difference = y * y_scale

    ax.set_ylim(bottom=bottom-y_difference, top=top+y_difference)
    ax.set_xlim(0, max_time + (max_time / num_versions))


def plot_elo_ratings(game_class, num_versions):
    tournament = read_ratings(game_class)
    version_times = load_elo_version_time(game_class)
    colors = sns.color_palette("Set1", n_colors=len(tournament), desat=.5)

    max_time = max([version_times[p] * num_versions for p, vt in version_times.items()])
    # Decide unit type.
    unit_label = "Seconds"
    unit_division = 1
    if max_time >= 60:
        unit_division *= 60
        max_time = max_time / 60
        unit_label = "Minutes"
    if max_time >= 60:
        unit_division *= 60
        max_time = max_time / 60
        unit_label = "Hours"

    ax = None
    for i, dic in enumerate(tournament):
        player_key = list(dic.keys())[0]
        player_info = dic[player_key]
        x = np.array(player_info["elo"])
        uncertainty = np.array([player_info["uncertainty+"], player_info["uncertainty-"]])
        data = x + uncertainty
        if player_key.startswith("ExIt"):
            # ExIt players.
            time = [(i+1)*version_times[player_key] / unit_division for i in range(num_versions)]
            ax = sns.tsplot(data=data, time=time, color=colors[i],
                            condition=player_key, marker="o", linestyle="-",
                            ax=ax)
        else:
            # Non-ExIt players.
            data = [[a, a] for a in data.tolist()]
            ax = sns.tsplot(data=data, time=[-9999, 9999], color=colors[i], linewidth=2,
                            condition=player_key, marker="", linestyle="--", ax=ax)

    ax.set(ylabel='Elo Rating', xlabel='Training ' + unit_label)
    plt.subplots_adjust(top=0.9 - len(tournament) * 0.04, bottom=0.15)
    add_y_margins_inside(ax, max_time, num_versions, y_scale=0.1)

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
