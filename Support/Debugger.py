
from Games.TicTacToe import TicTacToe


class TicTacToeDebugger:

    def debug_display_win_moves(self, player):
        self.debug_display_test(player=player, action_list=[0, 4, 1, 8])
        self.debug_display_test(player=player, action_list=[4, 1, 0, 7])
        self.debug_display_test(player=player, action_list=[8, 0, 7, 2])
        self.debug_display_test(player=player, action_list=[8, 0, 7, 5])
        self.debug_display_test(player=player, action_list=[4, 1, 6, 3])
        self.debug_display_test(player=player, action_list=[0, 1, 3, 7])

    def debug_display(self, player):
        self.debug_display_test(player=player, action_list=[])
        self.debug_display_test(player=player, action_list=[4])
        self.debug_display_test(player=player, action_list=[4, 1])
        self.debug_display_test(player=player, action_list=[4, 0])
        self.debug_display_test(player=player, action_list=[1, 4, 3])
        self.debug_display_test(player=player, action_list=[1, 4, 3, 5])
        self.debug_display_test(player=player, action_list=[1, 4, 5, 3])
        self.debug_display_test(player=player, action_list=[4, 0, 8, 5])

    @staticmethod
    def debug_display_test(player, action_list):
        print("")
        print("")
        game = TicTacToe()
        for a in action_list:
            game.advance(a)
        game.display()
        player.move(game)
        game.display()
