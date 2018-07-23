
from Misc.DiskHandler import load_model
import numpy as np


def play(game_class):

    while True:
        print("*** New game of " + game_class.__name__ + " ***")
        print("")

        state = game_class()

        while not state.is_game_over():
            state.display()
            print("(Player index to move is: " + str(state.turn) + ".")
            a = input("Action: ")
            try:
                a = int(a)
            except:
                print("Action is not a number")
                continue

            if a not in state.get_legal_moves():
                print("ERROR: Action is not legal!")
                print("")
                continue

            print("Action taken is: " + str(a))
            print("")
            state.advance(a)

        state.display()

        if state.winner == -1:
            print("Game is a DRAW!")
        else:
            print("Game is won by player: " + str(state.winner))
        print("")
        print("")
        input("Press enter to play another game: ")


def play_trained(game_class, player, version, search_time=None, always_explot=True):
    trained_model = load_model(
        game_name=game_class.__name__,
        ex_it_algorithm=player.ex_it_algorithm,
        iteration=version
    )
    player.ex_it_algorithm.apprentice.set_model(trained_model)

    while True:
        print("*** New game of " + game_class.__name__ + " ***")
        print("")

        a = input("Do you want to start? (\"y\" or \"n\"): ")
        if a != "y" and a != "n":
            print("ERROR: input not recognized! Input was: \'" + str(a) + "\' with type: " + str(type(a)))
            continue

        state = game_class()
        human_index = 0 if a == "y" else 1

        while not state.is_game_over():
            state.display()

            if state.turn == human_index:
                print("(Player index to move is: " + str(state.turn) + ".")
                a = input("Action: ")
                try:
                    a = int(a)
                except:
                    print("Action is not a number")
                    continue

                if a not in state.get_legal_moves():
                    print("ERROR: Action is not legal!")
                    print("")
                    continue
            else:
                if search_time is not None:
                    a, a_best, v = player.expert.search(
                        state, player.apprentice, player.search_time, always_explot
                    )
                else:
                    a, pi_pred, v_pred = player.move(state, randomness=False)
                    print("pi_pred =", pi_pred)
                    print("v_pred =", v_pred)
                    print("Action taken is: " + str(a))
                    print("")
                    continue

            print("Action taken is: " + str(a))
            print("")
            state.advance(a)

        state.display()

        if state.winner == -1:
            print("Game is a DRAW!")
        else:
            print("Game is won by player: " + str(state.winner))
        print("")
        print("")
        input("Press enter to play another game: ")
