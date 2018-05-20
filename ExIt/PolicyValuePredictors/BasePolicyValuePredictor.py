

class BasePolicyValuePredictor:
    """
    Base class used for policy and evaluation predictors.
    """

    predictor = None

    def pred_value(self, feature_vector):
        prob_and_eval = self.predictor.predict(feature_vector)
        return prob_and_eval[-1]

    def pred_prob(self, feature_vector):
        prob_and_eval = self.predictor.predict(feature_vector)
        return prob_and_eval[:-1]
