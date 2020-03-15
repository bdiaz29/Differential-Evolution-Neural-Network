import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
import numpy as np
from numpy import asfarray
from numpy import asarray_chkfinite
import datetime
from scipy.optimize import rosen, differential_evolution
from random import random
from random import seed
import time
import math
import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.backend import manual_variable_initialization

manual_variable_initialization(True)


# bounds 7 neural the size of the array
# and then the learning rate/number of epochs
def funcc(x):
    x = asarray_chkfinite(x)
    size = len(x) - 6
    extra1 = x[size]
    epoc = int(extra1)
    extra2 = x[size + 1]
    extra3 = x[size + 2]
    extra5 = x[size + 3]
    extra6 = x[size + 4]
    extra7 = x[size + 5]
    OP = int(extra7)
    LO = int(extra3)
    batchzi = int(extra2)
    alp = (extra1 - int(extra1))

    losses = [' '] * 16

    losses[0] = "mean_squared_error"
    losses[1] = "mean_absolute_error"
    losses[2] = "mean_absolute_percentage_error"
    losses[3] = "mean_squared_logarithmic_error"
    losses[4] = "squared_hinge"
    losses[5] = 'hinge'
    losses[6] = 'categorical_hinge'
    losses[7] = 'logcosh'
    losses[8] = 'huber_loss'
    losses[9] = 'sparse_categorical_crossentropy'
    losses[10] = 'binary_crossentropy'
    losses[11] = 'kullback_leibler_divergence'
    losses[12] = 'poisson'

    var1 = extra5
    var2 = extra6
    learningrate = var2 - int(var2)
    acti = Activation('linear')

    opti = optimizers.SGD(lr=(learningrate * .1), decay=0, momentum=0.0, nesterov=False)
    if OP == 0:
        opti = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
    elif OP == 1:
        opti = optimizers.SGD(learning_rate=learningrate, momentum=0.0, nesterov=False)
    elif OP == 2:
        opti = optimizers.SGD(learning_rate=learningrate, momentum=0.0, nesterov=True)
    elif OP == 3:
        opti = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    elif OP == 4:
        opti = optimizers.RMSprop(learning_rate=learningrate, rho=0.9)
    elif OP == 5:
        opti = optimizers.Adagrad(learning_rate=0.01)
    elif OP == 7:
        opti = optimizers.Adagrad(learning_rate=learningrate)
    elif OP == 8:
        opti = optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    elif OP == 9:
        opti = optimizers.Adadelta(learning_rate=learningrate, rho=0.95)
    elif OP == 10:
        opti = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif OP == 11:
        opti = optimizers.Adam(learning_rate=learningrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif OP == 12:
        opti = optimizers.Adam(learning_rate=learningrate, beta_1=var1, beta_2=var2, amsgrad=False)
    elif OP == 13:
        opti = optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif OP == 14:
        opti = optimizers.Adamax(learning_rate=learningrate, beta_1=0.9, beta_2=0.999)
    elif OP == 15:
        opti = optimizers.Adamax(learning_rate=learningrate, beta_1=var1, beta_2=var2)
    elif OP == 16:
        opti = optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    elif OP == 17:
        opti = optimizers.Nadam(learning_rate=learningrate, beta_1=0.9, beta_2=0.999)
    elif OP == 18:
        opti = optimizers.Nadam(learning_rate=learningrate, beta_1=var1, beta_2=var2)
    else:
        opti = optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

    if math.isnan(max(x)):
        x = np.full((size + 1, 1), 1)
    # n = x[size] + 1
    leak = [None] * size
    for i in range(size):
        leak[i] = (x[i] - int(x[i])) * .3
    leaky = np.mean(leak)
    leaky = leaky - int(leaky)
    nn_length = 0
    # determine length of nn
    for l in range(size):
        if x[l] >= 1:
            nn_length = nn_length + 1
    nn_create = [None] * nn_length
    nn_create_leak = [None] * nn_length
    nn_length = 0
    for cr in range(size):
        if x[cr] >= 1:
            nn_create[nn_length] = int(x[cr])
            nn_create_leak[nn_length] = leak[cr]
            nn_length = nn_length + 1
    # create training and testing data
    # first seed random number generator with system time
    seed(5)
    # initialize training and testing arrays
    # testing_range = 250
    # training_range = batchzi

    # f_testing_inputs = np.zeros((testing_range, 2))
    # f_testing_outputs = np.zeros((testing_range, 1))

    # training data
    global f_training_inputs
    global f_training_outputs
    ff_training_inputs = f_training_inputs[:batchzi]
    ff_training_outputs = f_training_outputs[:batchzi]

    global fr_look
    # testing data
    global testing_range
    global f_testing_inputs
    global f_testing_outputs
    global f_look
    # set random seed
    seed(1)
    # create neural network
    model = Sequential()
    # the first layer
    # model.add(Dense(2, input_dim=2, activation=acti, kernel_initializer=tf.keras.initializers.glorot_normal(seed=0)))
    # model.add(acti)
    # intermediate layers if any
    nn_length = nn_length
    for k in range(nn_length):
        v1 = nn_create_leak[k]
        v2 = nn_create_leak[k]
        V1 = v1 * 10000
        V2 = (V1 - int(V1)) * 10000
        A1 = int(V1) / 10000
        A2 = int(V2) / 10000
        acti = tf.keras.layers.ReLU(max_value=None, negative_slope=A1,
                                    threshold=A2)
        if k == 0:
            model.add(
                Dense(nn_create[k], input_dim=2, activation=acti,
                      kernel_initializer=tf.keras.initializers.glorot_normal(seed=0)))
        else:
            model.add(Dense(nn_create[k], activation=acti,
                            kernel_initializer=tf.keras.initializers.glorot_normal(seed=k + 1)))
    # the last layer
    model.add(Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.glorot_normal(seed=nn_length + 1)))
    # model.add(acti)
    # compile network
    adadelta = optimizers.Adadelta()
    adam = optimizers.Adam()
    passloss = losses[LO]
    sgd = optimizers.SGD(lr=(learningrate * .1), decay=0, momentum=0.0, nesterov=False)
    rprop = optimizers.RMSprop()
    model.compile(loss=passloss, optimizer=opti, metrics=['accuracy'])
    # train model
    model.fit(ff_training_inputs, ff_training_outputs, epochs=epoc, batch_size=batchzi, verbose=0)
    weights = model.get_weights()
    arc1 = model.to_yaml()
    # feed testing data through neural networks
    f_nn_out = model.predict(f_testing_inputs)
    if np.isnan(f_nn_out).any():
        return float("inf")
    insf = np.isneginf(f_nn_out)
    in1 = np.any(insf)
    insf = np.isposinf(f_nn_out)
    in1 = np.any(insf)
    in2 = np.any(insf)
    if in1 or in2:
        return float("inf")
    err = np.zeros((testing_range, 1))
    # from the testing nn outputs computer the average and max error
    f_error = (abs(f_nn_out[0] - f_testing_outputs[0]) / f_testing_outputs[0]) * 100
    f_nn_error_avg = abs(f_error)
    f_nn_error_max = abs(f_error)
    f_nn_error_min = abs(f_error)
    for c in range(1, testing_range):
        f_error = (abs(f_nn_out[c] - f_testing_outputs[c]) / f_testing_outputs[c]) * 100
        f_nn_error_avg = abs((f_error + f_nn_error_avg) * .5)
        f_nn_error_max = abs(max(f_error, f_nn_error_max))
        f_nn_error_min = abs(min(f_error, f_nn_error_min))
    # take the average of the average and max and returns as fitness
    # return as percent error
    # return 100 if nn died
    if f_nn_out.max() == 0:
        f_fitness = float("inf")
    else:
        f_fitness = abs(((f_nn_error_avg + f_nn_error_max) * .5))
    if math.isnan(f_fitness):
        f_fitness = float("inf")
    if f_fitness == float("-inf"):
        f_fitness = float("inf")
    global mun
    global countt
    f_fitness = model.evaluate(f_testing_inputs, f_testing_outputs, batch_size=batchzi, verbose=0)
    f_fitness = max(f_fitness)
    f_fitness = (2 * f_fitness + f_nn_error_max) / 3
    # f_fitness = f_fitness * f_nn_error_max
    # f_fitness = f_fitness * f_nn_error_max*100
    if f_fitness < .0000000000000000000000000000000000001:
        a_a = float(f_fitness)
        b_b = a_a.__round__(5)
        c_c = str(b_b)
        d_d = c_c.replace(".", "DOT")
        name1 = d_d
        countt = countt + 1
        name2 = 'Weights_model.h5'
        name4 = 'archi.txt'
        name3 = name1 + name2
        name5 = name1 + name4
        # file_k = open(name5, "w+")
        # model.save_weights(name3)
        # file_k.write(arc1)
        # file_k.close()
    if f_fitness < mun:
        now = datetime.datetime.now()

        mun = f_fitness
        file1 = open("progress.txt", "a")
        file1.write(str(x))
        file1.write(" avg:")
        file1.write(str(f_nn_error_avg))
        file1.write(" max:")
        file1.write(str(f_nn_error_max))
        file1.write(" fitness:")
        file1.write(str(f_fitness))
        file1.write(" completed:")
        file1.write(str(now))
        file1.write("\n")
        file1.close()

    del model
    return f_fitness


###################################################

testing_range = 2500
training_range = 100

# testing_range = 100
# training_range = 100

t_range = int(math.sqrt(testing_range))
tr_range = int(math.sqrt(training_range))
f_testing_inputs = np.zeros((testing_range, 2))
f_testing_outputs = np.zeros((testing_range, 1))

f_look = np.zeros((testing_range, 3))
fr_look = np.zeros((training_range, 3))

f_training_inputs = np.zeros((training_range, 2))
f_training_outputs = np.zeros((training_range, 1))
# training data
for ttr in range(tr_range):
    for tir in range(tr_range):
        a = tir + 1
        b = ttr + 1
        c = math.sqrt((a * a) + (b * b))
        f_training_inputs[(ttr * tr_range) + tir] = (a, b)
        f_training_outputs[(ttr * tr_range) + tir] = c
        fr_look[(ttr * tr_range) + tir] = (a, b, c)
# testing data
for tt in range(t_range):
    for ti in range(t_range):
        a = ti + 1
        b = tt + 1
        c = math.sqrt((a * a) + (b * b))
        f_testing_inputs[(tt * t_range) + ti] = (a, b)
        f_testing_outputs[(tt * t_range) + ti] = c
        f_look[(tt * t_range) + ti] = (a, b, c)
mun = float('inf')
countt = 0
# print(funcc((14.93788355  ,9.33733632  , 5.95348638,474.21187539,363.35827283,
#   3.08766271  , 1.85206141 ,  3.12711794,  11.32997014)))
# print(funcc((9.23543184,16.19068103, 18.17606252,444.3437548,621.46146325,
#   2.00393233,1.30802165 ,3.09851539 ,13.68659251)))
# bounds = [(1, 20), (1, 20), (1, 20), (1, 625), (1, 625), (0, 4.9), (0, 4.9999), (0, 5), (0, 18.9)]
# bounds = [(7, 20), (1, 20), (1, 20), (20, training_range), (100, training_range), (0, 4.9), (0, 4.9999), (0, 5), (0, 18.9)]


#normal
bounds = [(1, 25), (1, 25), (1, 25), (1, training_range), (1, training_range), (0, 3.9), (0, 3.9999), (0, 5), (0, 18.9)]
# bounds = [(1, 25), (1, 25), (1, 25), (1, training_range), (1, 1.1), (0, 3.9), (0, 3.9999), (0, 5), (0, 18.9)]
#print(differential_evolution(funcc, bounds, maxiter=1000, polish=False, popsize=3))
print(differential_evolution(funcc, bounds, maxiter=50, polish=False, popsize=4))


