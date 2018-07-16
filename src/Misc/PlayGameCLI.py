

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
