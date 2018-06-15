

from Games.TicTacToe import TicTacToe
from Players.Players import *
import numpy as np
np.set_printoptions(suppress=True)


search_time = 0.05
num_iteration = 1000


def main():
    #game_handler = GameHandler(game=TicTacToe(), players=[RandomPlayer(), RandomPlayer()])
    #game_handler.start_game()

    #player = NnMinimaxPlayer()
    player = NnMctsPlayer()
    player.set_game(game_class=TicTacToe)
    player.start_ex_it(num_iteration=num_iteration,
                       randomness=True,
                       search_time=search_time)

    debug_display(player)


def debug_display(player):

    print("")
    print("")
    game = TicTacToe()
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(1)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(0)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(3)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(3)
    game.advance(5)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(5)
    game.advance(3)
    game.display()
    player.move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(0)
    game.advance(8)
    game.advance(5)
    game.display()
    player.move(game)
    game.display()


if __name__ == "__main__":
    #cProfile.run('main()')
    main()
