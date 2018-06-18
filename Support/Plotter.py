
import matplotlib.pyplot as plt

# TODO: Fix metafile.
import csv
def plot_test(metafile=None, file=None):

    with open('./Statistics/2018-06-18___02-20-59/0.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [[float(row["win"]), float(row["loss"]), float(row["draw"])] for row in reader]

        print(data)
        iterations = 1000
        wins, loses, draws = get_statistics(data=data, iterations=iterations)
        plt.plot(wins, label="Wins")
        plt.plot(loses, label="Loses")
        plt.plot(draws, label="Draws")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0.)
        plt.axis(ymin=0, ymax=1.0)
        plt.show()


def get_statistics(data, iterations):
    wins = []
    loses = []
    draws = []
    for v in data:
        wins.append(v[0] / iterations)
        loses.append(v[1] / iterations)
        draws.append(v[2] / iterations)
    return wins, loses, draws