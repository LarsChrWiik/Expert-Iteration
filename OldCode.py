

#optimizer = SGD(lr=0.5, momentum=0.0, decay=0.0)
#self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


"""

# rnd_prob = 0.0 if not add_randomness else 1.0 / state.num_actions
#rnd_prob = 0 if not add_randomness else log(1+x)
#rnd_prob = 0 if not add_randomness else 4**(-2*(i / num_iteration))
# rnd_prob = math.e ** (-5 * x) + min_rnd

"""


"""
if self.is_leaf():
    X = self.state.get_feature_vector(self.state.turn)
    legal_moves = self.state.get_legal_moves(self.state.turn)
    action_prob = predictor.pred_prob(X=X)
    for i, v in enumerate(action_prob):
        if i not in legal_moves:
            action_prob[i] = -1
    return np.argmax(action_prob)

"""



"""
@staticmethod
def normalize(array, lower, upper):

    # Check of array contain non-unique elements.
    if np.unique(array).size == 1:
        return np.array([upper for _ in range(len(array))])

    diff = upper - lower
    _min = array[0]
    _max = array[0]
    for i, v in enumerate(array):
        if v < _min:
            _min = v
        if v > _max:
            _max = v

    new_array = np.zeros(len(array), dtype=float)

    for i, v in enumerate(array):
        norm = (v - _min) / (_max - _min)
        new_array[i] = (diff * norm) + lower

    return new_array
"""





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