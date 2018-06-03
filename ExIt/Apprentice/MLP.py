
from ExIt.Apprentice.BaseApprentice import BaseApprentice
from keras.optimizers import SGD, Adagrad
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Input, PReLU
import numpy as np
import keras.backend as backend
from keras.initializers import RandomNormal


class MLP(BaseApprentice):
    """
    Multi Layer Perceptron (MLP) implementation.
    """

    model = None
    optimizer = None

    def init_model(self, input_fv_size, pi_size):
        """
        Bias is included.
        """
        evaluation_size = 1

        main_input = Input(shape=(input_fv_size,))                              # Input layer.
        x = Dense(32, activation='elu')(main_input)   # Hidden layer 1.
        x = Dense(32, activation='elu')(x)   # Hidden layer 2.
        pi = Dense(pi_size, activation='tanh', name='pi_output')(x)          # Output layer 1
        r = Dense(evaluation_size, activation='tanh', name='r_output')(x)       # Output layer 2

        self.model = Model(inputs=[main_input], outputs=[pi, r])
        self.model.summary()

        #self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        loss_weights_pi = pi_size/(pi_size+1)
        loss_weights_r = 1/(pi_size+1)
        optimizer = SGD(lr=0.2, momentum=0.0, decay=0.0)
        self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizer,
                           loss={'pi_output': 'mean_squared_error', 'r_output': 'mean_squared_error'},
                           loss_weights={'pi_output': loss_weights_pi, 'r_output': loss_weights_r})

    def train(self, X, Y_pi, Y_r):
        self.model.fit(x=np.array(X), y=[np.array(Y_pi), np.array(Y_r)])

    def pred_eval(self, X):
        """
        Return float representing state evaluation.
        """
        return self.model.predict(x=np.array([X]))[1][0][0]

    def pred_prob(self, X):
        """
        Return array of action probabilities.
        """
        return self.model.predict(x=np.array([X]))[0][0]

    def set_lr(self, new_lr):
        backend.set_value(self.optimizer.lr, new_lr)
