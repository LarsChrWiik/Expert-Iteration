
from Games.TicTacToe import TicTacToe
from Players.BasePlayers import BaseExItPlayer


def debug_display_win_moves(player):
    debug_display_ttt(player=player, action_list=[0, 4, 1, 8])
    debug_display_ttt(player=player, action_list=[4, 1, 0, 7])
    debug_display_ttt(player=player, action_list=[8, 0, 7, 2])
    debug_display_ttt(player=player, action_list=[8, 0, 7, 5])
    debug_display_ttt(player=player, action_list=[4, 1, 6, 3])
    debug_display_ttt(player=player, action_list=[0, 1, 3, 7])


def debug_display_general_moves(player):
    debug_display_ttt(player=player, action_list=[])
    debug_display_ttt(player=player, action_list=[4])
    debug_display_ttt(player=player, action_list=[4, 1])
    debug_display_ttt(player=player, action_list=[4, 0])
    debug_display_ttt(player=player, action_list=[1, 4, 3])
    debug_display_ttt(player=player, action_list=[1, 4, 3, 5])
    debug_display_ttt(player=player, action_list=[1, 4, 5, 3])
    debug_display_ttt(player=player, action_list=[4, 0, 8, 5])


def debug_display_ttt(player, action_list):
    print("")
    print("")
    game = TicTacToe()
    for a in action_list:
        game.advance(a)
    game.display()
    # Do the optimal move. (No randomness).
    player.move(game, print_info=True, randomness=False)
    game.display()
