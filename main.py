
from Games.TicTacToe import TicTacToe
from Games.GameHandler import GameHandler
from Players.RandomPlayer import RandomPlayer


def main():

    handler = GameHandler(
        game=TicTacToe(),
        players=[RandomPlayer(), RandomPlayer()]
    )

    handler.start_game()


    """
    ttt1 = TicTacToe()
    ttt1.advance(player_index=1, action_index=4)
    ttt = ttt1.get_state_copy()
    ttt.advance(player_index=2, action_index=1)
    ttt.advance(player_index=1, action_index=0)
    ttt.advance(player_index=2, action_index=8)
    ttt.advance(player_index=1, action_index=3)
    ttt.advance(player_index=2, action_index=5)
    ttt.advance(player_index=1, action_index=6)
    ttt.display()
    print(ttt.winner)
    """



if __name__ == "__main__":
    main()
