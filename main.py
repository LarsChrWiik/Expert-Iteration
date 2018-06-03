
from Games.TicTacToe import TicTacToe
from Players.MinimaxMlpPlayer import MinimaxMlpPlayer


def main():
    #game_handler = GameHandler(game=TicTacToe(), players=[RandomPlayer(), RandomPlayer()])
    #game_handler.start_game()

    player = MinimaxMlpPlayer(minimax_depth=2)
    player.start_exIt(game_class=TicTacToe, num_iteration=10000, randomness=True)

    print("")
    print("")
    game = TicTacToe()
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(1)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(0)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(3)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(3)
    game.advance(5)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(1)
    game.advance(4)
    game.advance(5)
    game.advance(3)
    game.display()
    player.make_expert_move(game)
    game.display()

    print("")
    print("")
    game = TicTacToe()
    game.advance(4)
    game.advance(0)
    game.advance(8)
    game.advance(5)
    game.display()
    player.make_expert_move(game)
    game.display()



if __name__ == "__main__":
    #cProfile.run('main()')
    main()
