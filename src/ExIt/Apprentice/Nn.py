
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

    n_neurons = 2**8
    n_layers = 3
    dropout_rate = None
    v_size = 1
    regularisation_strength = 0.01
    optimizer = Adam()

    def __init__(self):
        self.model = None
        self.optimizer = Nn.optimizer
        self.policy_network = None
        self.value_network = None

    def init_model(self, input_fv_size, pi_size):
        # Input layer.
        input = Input(shape=(input_fv_size,))

        # Hidden layers.
        x = input
        for _ in range(Nn.n_layers):
            x = Dense(Nn.n_neurons, kernel_regularizer=l2(Nn.regularisation_strength))(x)
            x = BatchNormalization()(x)
            x = Activation("elu")(x)
            if Nn.dropout_rate is not None and Nn.dropout_rate != 0.0:
                x = Dropout(rate=Nn.dropout_rate)(x)

        # Output layer pi = Action probability.
        p = Dense(pi_size, kernel_regularizer=l2(Nn.regularisation_strength))(x)
        p = BatchNormalization()(p)
        p = Activation("softmax", name='p_output')(p)
        # Output layer v = state evaluation
        v = Dense(Nn.v_size, kernel_regularizer=l2(Nn.regularisation_strength))(x)
        v = BatchNormalization()(v)
        v = Activation("tanh", name='v_output')(v)

        model = Model(inputs=[input], outputs=[p, v])

        model.compile(
            optimizer=self.optimizer,
            loss={
                'p_output': categorical_crossentropy,
                'v_output': mean_squared_error
            }
        )

        self.model = model
        self.policy_network = Model(inputs=[input], outputs=[p])
        self.value_network = Model(inputs=[input], outputs=[v])

    def train(self, X_s, Y_p, Y_v):
        _, policy_loss, value_loss = self.model.train_on_batch(
            x=np.array(X_s),
            y=[np.array(Y_p), np.array(Y_v)]
        )
        return policy_loss, value_loss

    def pred_v(self, X):
        """ Return float representing state evaluation """
        return self.value_network.predict(x=np.array([X]))[0][0]

    def pred_p(self, X):
        """ Return array of action probabilities """
        return self.policy_network.predict(x=np.array([X]))[0]

    def set_lr(self, new_lr):
        """ Set new learning rate """
        K.set_value(self.optimizer.lr, new_lr)

    def get_lr(self):
        """ Return the learning rate """
        return K.get_value(self.optimizer.lr)

    def set_model(self, trained_model):
        """ Set the model to a trained model """
        self.model = trained_model
        self.policy_network = Model(inputs=self.model.input, outputs=self.model.output[0])
        self.value_network = Model(inputs=self.model.input, outputs=self.model.output[1])
