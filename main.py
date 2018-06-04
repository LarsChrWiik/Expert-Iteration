
from Games.TicTacToe import TicTacToe
from Players.MinimaxMlpPlayer import MinimaxMlpPlayer
from Games.GameHandler import GameHandler
from Players.RandomPlayer import RandomPlayer


def main():
    #game_handler = GameHandler(game=TicTacToe(), players=[RandomPlayer(), RandomPlayer()])
    #game_handler.start_game()

    player = MinimaxMlpPlayer(minimax_depth=2)
    player.start_ex_it(game_class=TicTacToe, num_iteration=1000, randomness=True)

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
