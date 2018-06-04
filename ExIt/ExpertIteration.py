
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.BaseGame import BaseGame
from ExIt.DataSet import DataSet
from random import uniform
from random import choice as rnd_choice


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()

    def calculate_best_action(self, state: BaseGame):
        return self.expert.start(state=state, predictor=self.apprentice)

    def start_ex_it(self, game_class, num_iteration, add_randomness: bool):
        """
        Start Expert Iteration to master the given game.
        This process is time consuming.
        """
        self.apprentice.init_model(input_fv_size=game_class().fv_size,
                                   pi_size=game_class().num_actions)

        for i in range(num_iteration):
            # TODO: Move to NN maybe?
            if i == int(num_iteration/4):
                self.apprentice.set_lr(new_lr=0.1)
            elif i == int(num_iteration/2):
                self.apprentice.set_lr(new_lr=0.05)
            elif i == int(num_iteration/(4/3)):
                self.apprentice.set_lr(new_lr=0.01)

            state = game_class()

            # TODO: Ask Spyros.
            x = (1 - (i / num_iteration))
            #rnd_prob = 0 if not randomness else x
            rnd_prob = 0 if not add_randomness else 0.1
            # rnd_prob = 0.0 if not randomness else 1.0 / state.num_actions
            #rnd_prob = 0 if not randomness else log(1+x)
            #rnd_prob = 0 if not randomness else 4**(-2*(i / num_iteration))

            X = state.get_feature_vector(state.turn)
            while not state.has_finished():
                # TODO: Maybe remove return.
                did_random = self.ex_it_state(state=state, rnd_prob=rnd_prob)
                # TODO: Maybe remove.
                #if i > 500 and did_random:
                 #   rnd_prob = rnd_prob / 2
            print("*** Iteration = " + str(i) + " ***")
            print("randomness = " + str(rnd_prob))
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

    def ex_it_state(self, state: BaseGame, rnd_prob: float):

        # Ask expert for the best move.
        action_index, r, pi_update = self.expert.start(state=state, predictor=self.apprentice)
        # Add sample.
        self.data_set.add_sample(state=state, pi_update=pi_update, r=r)

        did_random = False
        if uniform(0, 1) < rnd_prob:
            did_random = True
            a = state.get_legal_moves(state.turn)
            action_index = rnd_choice(a)

            # TODO: Fix later.
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

        # TODO: Maybe remove.
        return did_random
