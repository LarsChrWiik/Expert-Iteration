
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.BaseGame import BaseGame
from ExIt.DataSet import DataSet
from random import uniform
from random import choice as rnd_choice


class ExpertIteration:
    """
    Expert Iteration.
    """

    # NN.
    apprentice = None

    # MCTS or AB-search.
    expert = None

    data_set = DataSet()

    game_class = None

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert

    def calculate_best_action(self, state: BaseGame):
        """
        This is used after the Expert Iteration has completed.

        :return: int, action index.
        """
        return self.expert.start(state=state, predictor=self.apprentice)

    def start_exIt(self, game_class, num_iteration, randomness: bool):
        """
        Start Expert Iteration to master the given game.
        This process might take some time.
        """
        self.apprentice.init_model(input_fv_size=game_class().fv_size,
                                   pi_size=game_class().num_actions)
        self.game_class = game_class

        for i in range(num_iteration):
            if i == int(num_iteration/4):
                self.apprentice.set_lr(new_lr=0.1)
            elif i == int(num_iteration/2):
                self.apprentice.set_lr(new_lr=0.05)
            elif i == int(num_iteration/(4/3)):
                self.apprentice.set_lr(new_lr=0.01)

            state = game_class()

            # TODO: fix.
            x = (1 - (i / num_iteration))
            #t = 0 if not randomness else x
            t = 0 if not randomness else 0.1
            # t = 0.0 if not randomness else 1.0 / state.num_actions
            #t = 0 if not randomness else log(1+x)
            #t = 0 if not randomness else 4**(-2*(i / num_iteration))

            X = state.get_feature_vector(state.turn)
            while not state.has_finished():
                did_random = self.exIt_state(state=state, randomness=t)
                #if i > 500 and did_random:
                 #   t = t / 2
            print("*** Iteration = " + str(i) + " ***")
            print("randomness = " + str(t))
            print("    r = ", self.data_set.r_array)
            # TODO: Check this.
            self.data_set.update_reward_hard(state)
            print("r_new = ", self.data_set.r_array)

            # Train the Policy and Evaluation Predictor.
            s_array, pi_array, r_array = self.data_set.extract_data()
            if len(s_array) > 0:
                print("    pi = ", self.apprentice.pred_prob(X=X))
                print("pi_new = ", pi_array[0])
                print("r_start= ", self.apprentice.pred_eval(X=X))
                print("")
                self.apprentice.train(X=s_array, Y_pi=pi_array, Y_r=r_array)
            print("")
            print("")

    # TODO: change name of randomness. = t.
    def exIt_state(self, state: BaseGame, randomness: float):

        # Ask expert for the best move.
        action_index, r, pi_update = self.expert.start(state=state, predictor=self.apprentice)
        # Add sample.
        self.data_set.add_sample(state=state, pi_update=pi_update, r=r)

        did_random = False
        if uniform(0, 1) < randomness:
            did_random = True
            a = state.get_legal_moves(state.turn)
            action_index = rnd_choice(a)

            """
            if uniform(0, 1) > randomness:
                # Random move based on action probabilities.
                X = state.get_feature_vector(state.turn)

                p = self.apprentice.pred_prob(X)
                p = [x for i, x in enumerate(p) if i in a]
                p = NodeMiniMax.normalize(array=p, lower=0.25, upper=0.75)
                s = sum(p)
                p = np.array([x / s for x in p])

                action_index = random.choice(a=a, size=1, p=p)[0]
            else:
                if 0.3 > randomness:
                    self.data_set.clear()
                # Completely random move.
                action_index = rnd_choice(a)
            """

        state.advance(action_index=action_index)

        return did_random
