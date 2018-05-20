
from Games.TicTacToe import TicTacToe
from Games.GameHandler import GameHandler
from Players.RandomPlayer import RandomPlayer
from ExIt.PolicyImprovers.MiniMax.MiniMax import MiniMax
from ExIt.PolicyValuePredictors.MLP import MLP

def main():
    #game_handler = GameHandler(game=TicTacToe(), players=[RandomPlayer(), RandomPlayer()])
    #game_handler.start_game()

    game = TicTacToe()
    game.advance(1)
    game.display()

    nn = MLP()
    minimax = MiniMax(depth=9)
    action_index = minimax.search_and_store(game=game, predictor=nn)
    print(action_index)

if __name__ == "__main__":
    main()
