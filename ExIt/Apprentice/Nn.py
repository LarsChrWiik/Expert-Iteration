
from ExIt.Apprentice.BaseApprentice import BaseApprentice
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Activation, Dropout
import numpy as np
import keras.backend as K
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.regularizers import l2


def entropy(y):
    # Clip to avoid Zeros.
    y = K.clip(y, 1e-20, 1.0)
    return -K.sum(y * K.log(y))


def custom_loss(y_true, y_pred, beta=0.001):
    return categorical_crossentropy(y_true, y_pred) - beta * entropy(y_pred)


class Nn(BaseApprentice):
    """ Deep neural network implementation (Apprentice) """

    def __init__(self):
        self.model = None
        self.optimizer = None

    def init_model(self, input_fv_size, pi_size):
        # Input layer.
        x = Input(shape=(input_fv_size,))

        # Hidden layer 1.
        h1 = Dense(256, kernel_regularizer=l2())(x)
        h1 = Activation("elu")(h1)
        #h1 = BatchNormalization()(h1)
        h1 = Dropout(rate=0.2)(h1)

        # Hidden layer 2
        h2 = Dense(256, kernel_regularizer=l2())(h1)
        h2 = Activation("elu")(h2)
        #h2 = BatchNormalization()(h2)
        h2 = Dropout(rate=0.2)(h2)

        # Output layer pi = Action probability.
        pi = Dense(pi_size, activation='softmax', name='pi_output', kernel_regularizer=l2())(h2)
        # Output layer v = state evaluation
        v = Dense(1, activation='linear', name='v_output', kernel_regularizer=l2())(h2)

        model = Model(inputs=[x], outputs=[pi, v])
        model.summary()

        optimizer = Adam()
        model.compile(
            optimizer=optimizer,
            loss={
                'pi_output': categorical_crossentropy,
                'v_output': mean_squared_error
            }
        )

        self.optimizer = optimizer
        self.model = model

    def train(self, X, Y_pi, Y_v):
        self.model.fit(x=np.array(X), y=[np.array(Y_pi), np.array(Y_v)])

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
