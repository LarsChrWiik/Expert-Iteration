
from ExIt.Apprentice.BaseApprentice import BaseApprentice
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Input, LeakyReLU
import numpy as np
import keras.backend as K
from keras.losses import mean_squared_error, categorical_crossentropy as cc
# from keras.regularizers import l2


def entropy(y):
    # Clip to avoid Zeros.
    y = K.clip(y, 1e-20, 1.0)
    return -K.sum(y * K.log(y))


def custom_loss(y_true, y_pred, beta=0.001):
    return cc(y_true, y_pred) + beta * entropy(y_pred)


class Nn(BaseApprentice):
    """ Multi Layer Perceptron (MLP) implementation """

    model = None
    optimizer = None
    eval_size = 1

    def init_model(self, input_fv_size, pi_size):

        x = Input(shape=(input_fv_size,))                                   # Input layer
        h1 = Dense(32)(x)                                                   # Hidden layer 1
        h1 = LeakyReLU()(h1)                                                # Hidden layer 1 (Act)
        h2 = Dense(32)(h1)                                                  # Hidden layer 2
        h2 = LeakyReLU()(h2)                                                # Hidden layer 2 (Act)
        pi = Dense(pi_size, activation='softmax', name='pi_output')(h2)     # Output layer 1
        v = Dense(self.eval_size, activation='linear', name='r_output')(h2) # Output layer 2

        self.model = Model(inputs=[x], outputs=[pi, v])
        self.model.summary()

        # TODO: Maybe remove later.
        loss_weights_pi = pi_size/(pi_size+1)
        loss_weights_r = 1/(pi_size+1)
        loss_weights = {
            'pi_output': loss_weights_pi,
            'r_output': loss_weights_r
        }

        optimizer = Adam()
        self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizer,
                           loss={
                               'pi_output': custom_loss,
                               'r_output': mean_squared_error
                           },
                           loss_weights=None)

    def train(self, X, Y_pi, Y_r):
        self.model.fit(x=np.array(X), y=[np.array(Y_pi), np.array(Y_r)])

    def pred_eval(self, X):
        """ Return float representing state evaluation """
        return self.model.predict(x=np.array([X]))[1][0][0]

    def pred_prob(self, X):
        """ Return array of action probabilities """
        return self.model.predict(x=np.array([X]))[0][0]

    def set_lr(self, new_lr):
        """ Set new learning rate """
        K.set_value(self.optimizer.lr, new_lr)

    def get_lr(self):
        """ Return the learning rate """
        return K.get_value(self.optimizer.lr)
