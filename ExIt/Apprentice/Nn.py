
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
        h1 = BatchNormalization()(h1)
        h1 = Activation("elu")(h1)
        h1 = Dropout(rate=0.1)(h1)

        # Hidden layer 2
        h2 = Dense(256, kernel_regularizer=l2())(h1)
        h2 = BatchNormalization()(h2)
        h2 = Activation("elu")(h2)
        h2 = Dropout(rate=0.1)(h2)

        # Output layer pi = Action probability.
        pi = Dense(pi_size, activation='linear', kernel_regularizer=l2())(h2)
        #pi = BatchNormalization()(pi)
        pi = Activation("softmax", name='pi_output')(pi)
        # Output layer v = state evaluation
        v = Dense(1, activation='linear', kernel_regularizer=l2())(h2)
        #v = BatchNormalization()(v)
        v = Activation("tanh", name='v_output')(v)

        model = Model(inputs=[x], outputs=[pi, v])
        model.summary()

        optimizer = Adam()
        model.compile(
            optimizer=optimizer,
            loss={
                'pi_output': categorical_crossentropy,
                'v_output': mean_squared_error
            },
            loss_weights={
                'pi_output': 1,
                'v_output': 1
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



### Alternative Optimizer, might or might not work better than Adam

from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K


class NormalizedSGD(Optimizer):
    def __init__(
            self,
            lr: float=0.01,
            lr_update: float=0.001,
            lr_max: float=0.001,
            lr_min: float=1e-5,
            lr_force: float=0.0,
            norm: str = 'max',
            **kwargs
    ) -> None:
        """NormalizedSGD constructor

        :param lr: initial learning rate
        :param lr_update: relative learning rate update step. New lr is computed
            approximately as lr' = lr * (1 - lr_update * cos(a)), a is the
            angle between gradient in the k-th step and the k-1 step.
        :param lr_max: max value of lr
        :param lr_min: min value of lr
        :param lr_force: relative force lr to increase in consecutive steps.
            This is achieved following update:
                    lr" = lr' + lr_force * lr_update * lr'

        """
        super(NormalizedSGD, self).__init__(**kwargs)

        if norm not in ['max', 'l2']:
            raise ValueError('Unexpected norm type `{norm}`.')

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')

        self.lr = lr
        self.lr_update = lr_update
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_force = lr_force
        self.norm = norm
        self.learning_rates = None
        self.old_grads = None

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self.lr
        lr_update = self.lr_update
        lr_force = self.lr_force

        shapes = [K.int_shape(p) for p in params]
        learning_rates = [K.variable(lr) for _ in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.learning_rates = learning_rates
        self.old_grads = old_grads

        self.weights = [self.iterations]
        for p, g, l, old_g in zip(params, grads, learning_rates, old_grads):

            if self.norm == 'max':
                g_max = K.max(K.abs(g), axis=None, keepdims=True)
                denominator = K.epsilon() + g_max
                g_step_normed = g / denominator
            else:
                g_step_normed = K.l2_normalize(g)

            # update parameters with SGD
            new_p = p - l * g_step_normed
            self.updates.append(K.update(p, new_p))

            # update learning rate
            g_normed = K.l2_normalize(g)
            old_g_normed = K.l2_normalize(old_g)

            lr_change = - lr_update * K.sum(g_normed * old_g_normed)
            new_lr = l * (1 - lr_change) + lr_force * l * lr_update
            new_lr = K.clip(new_lr, self.lr_min, self.lr_max)

            self.updates.append(K.update(l, new_lr))
            self.updates.append(K.update(old_g, g))

        return self.updates

    def get_config(self):
        config = {
            'lr': self.lr,
            'lr_update': self.lr_update,
            'lr_max': self.lr_max,
            'lr_min': self.lr_min,
            'lr_force': self.lr_force,
            'norm': self.norm
        }
        base_config = super(NormalizedSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
