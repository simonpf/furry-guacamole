import numpy as np
import matplotlib
import copy
import os
import pickle

# Keras Imports
import keras
from keras.models     import Sequential
from keras.layers     import Dense, Activation, Dropout
from keras.optimizers import SGD

################################################################################
# Loss Functions
################################################################################

import keras.backend as K

def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:

    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)

    Where I is the indicator function.
    """
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)

def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:,0]), taus[0])
    for i,tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true),
                                   K.flatten(y_pred[:,i+1]),
                                   tau)
    return e

class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.

    Attributes:

        quantiles: List of quantiles that should be estimated with
                   this loss function.

    """
    def __init__(self, quantiles):
        self.__name__ = "Quantile Loss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

################################################################################
# Keras Interface Classes
################################################################################

class TrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:

        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """
    def __init__(self, x_train, x_mean, x_sigma, y_train, sigma_noise, batch_size):
        self.bs = batch_size

        self.x_train  = x_train
        self.x_mean   = x_mean
        self.x_sigma  = x_sigma
        self.y_train  = y_train
        self.sigma_noise = sigma_noise

        self.indices = np.random.permutation(x_train.shape[0])
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        inds = np.arange(self.i * self.bs, (self.i + 1) * self.bs)
        inds = self.indices[inds % self.x_train.shape[0]]

        x_batch  = self.x_train[inds,:]
        if (self.sigma_noise):
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        x_batch  = (x_batch - self.x_mean) / self.x_sigma
        y_batch  = self.y_train[inds]

        self.i = self.i + 1
        return (x_batch, y_batch)

class ValidationGenerator:
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.

    Attributes:

        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """
    def __init__(self, x_val, x_mean, x_sigma, y_val, sigma_noise):
        self.x_val  = x_val
        self.x_mean   = x_mean
        self.x_sigma  = x_sigma

        self.y_val  = y_val

        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val = self.x_val
        if self.sigma_noise:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        x_val = (x_val - self.x_mean) / self.x_sigma
        return (x_val, self.y_val)

class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.

    Attributes:

        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.

    """
    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay   = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps

    def on_train_begin(self, logs={}):
        self.losses = []
        self.convergence_steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.convergence_steps = self.convergence_steps + 1
        else:
            self.convergence_steps = 0.0
        if self.convergence_steps > 5:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(self.model.optimizer.lr, lr / 2.0)
            self.convergence_steps = 0.0
            print("\n Reduced learning rate to " + str(lr))

            if lr < 1e-8:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

################################################################################
# QRNN
################################################################################

default_settings = {"batch_size" : 512,
                    "lr_start" : 1e-1,
                    "lr_decay" : 2.0,
                    "lr_minimum" : 1e-8,
                    "epochs" : 200,
                    "convergence_epochs" : 10,
                    "train_split" : 0.9}

class QRNN:

    def __init__(self, input_dim, quantiles, depth, width, activation = "relu"):
        self.input_dim   = input_dim
        self.quantiles  = np.array(quantiles)
        self.depth      = depth
        self.width      = width
        self.activation = activation

        self.model = Sequential()
        self.model.add(Dense(input_dim = input_dim,
                             units = width,
                             activation = activation))
        for i in range(depth - 1):
            self.model.add(Dense(units = width,
                                activation = activation))
        self.model.add(Dense(units = len(quantiles),
                             activation = None))

    def fit(self, x_train, y_train, sigma_noise = None, settings = default_settings):

        x_mean  = np.mean(x_train, axis = 0, keepdims = True)
        x_sigma = np.std(x_train, axis = 0, keepdims = True)
        self.x_mean  = x_mean
        self.x_sigma = x_sigma

        n = x_train.shape[0]
        n_train = round(settings["train_split"] * n)
        n_val = n - n_train
        inds  = np.random.permutation(n)
        x_val   = x_train[inds[n_train:], :]
        y_val   = y_train[inds[n_train:]]
        x_train = x_train[inds[:n_train], :]
        y_train = y_train[inds[:n_train]]

        optimizer = SGD(lr = settings["lr_start"])
        loss = QuantileLoss(self.quantiles)
        self.custom_objects = {loss.__name__ : loss}
        self.model.compile(loss = loss,
                           optimizer = optimizer)

        training_generator   = TrainingGenerator(x_train,
                                                 self.x_mean,
                                                 self.x_sigma,
                                                 y_train,
                                                 sigma_noise,
                                                 settings["batch_size"])
        validation_generator = ValidationGenerator(x_val,
                                                   self.x_mean,
                                                   self.x_sigma,
                                                   y_val,
                                                   sigma_noise)
        lr_callback = LRDecay(self.model,
                              settings["lr_decay"],
                              settings["lr_minimum"],
                              settings["convergence_epochs"])

        n_train = x_train.shape[0]
        self.model.fit_generator(training_generator,
                                 steps_per_epoch = n_train // settings["batch_size"],
                                 epochs = settings["epochs"],
                                 validation_data = validation_generator,
                                 validation_steps = 1,
                                 callbacks = [lr_callback])

    def predict(self, x):
        return self.model.predict((x - self.x_mean) / self.x_sigma)

    def crps(self, x, y):

        y_pred = np.zeros(x.shape[0], self.quantiles.size + 2)
        y_pred[:, 1:-1] = self.predict(x)
        y_pred[:, 0] = 2.0 * y_pred[:, 1] - y_pred[:, 2]
        y_pred[:, -1] = 2.0 * y_pred[:, -2] - y_pred[:, -3]

        ind = zeros(y_pred.shape)
        ind[y_pred > y] = 1.0

        qs = np.zeros((1, self.quantiles.size + 2))
        qs[0, 1:-1] = self.quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0

        crps = np.trapz((qs - ind)**2.0, y_pred)

        return np.mean(crps)

    def save(self, path):

        f = open(path, "wb")
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]

        self.model_file = name + "_model"
        self.model.save(self.model_file)
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(path):
        f = open(path, "rb")
        qrnn = pickle.load(f)
        qrnn.model = keras.models.load_model(qrnn.model_file,
                                             qrnn.custom_objects)
        f.close()
        return qrnn

    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("model")
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
