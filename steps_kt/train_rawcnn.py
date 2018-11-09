#!/usr/bin/python3

##  Copyright (C) 2018 Idiap Research Institute
##  Authors:
##      S. Pavankumar Dubagunta
##      Vinayak Abrol
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras
import keras.backend as K
from keras.optimizers import SGD
from dataGeneratorSRaw import dataGenerator
from model_architecture import model_architecture
from compute_priors import compute_priors
import numpy
import sys
import os

if __name__ != '__main__':
    raise ImportError ('This script can only be run, and can\'t be imported')

if len(sys.argv) not in [8,9]:
    raise TypeError ('USAGE: train.py data_cv ali_cv data_tr ali_tr gmm_dir architecture dnn_dir')

data_cv = sys.argv[1]
ali_cv  = sys.argv[2]
data_tr = sys.argv[3]
ali_tr  = sys.argv[4]
gmm     = sys.argv[5]
arch    = sys.argv[6]
exp     = sys.argv[7]
init    = sys.argv[8] if len(sys.argv)==9 else None

## Learning parameters
learning = {'rate' : 0.1,
            'minEpoch' : 5,
            'lrScale' : 0.5,
            'batchSize' : 256,
            'lrScaleCount' : 18,
            'minValError' : 0.002}

os.makedirs (exp, exist_ok=True)
compute_priors (gmm, ali_tr, ali_cv)

trGen = dataGenerator (data_tr, ali_tr, gmm, learning['batchSize'])
cvGen = dataGenerator (data_cv, ali_cv, gmm, learning['batchSize'])

## Initialise learning parameters and models
s = SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=False)
## Model definition
numpy.random.seed(512)
m = model_architecture (arch, trGen.inputFeatDim, trGen.outputFeatDim)

if init:
    ## Initial model
    m_init = keras.models.load_model(init)
    
    ## Set weights from initial model
    layer_names = [layer.name for layer in m.layers if layer.name.startswith('conv1d')] + ['dense_1']
    for name in layer_names:
        m.get_layer(name).set_weights(m_init.get_layer(name).get_weights())

## Initial training
m.compile(loss='sparse_categorical_crossentropy', optimizer=s, metrics=['accuracy'])
print ('Learning rate: %f' % learning['rate'])
h = [m.fit_generator (trGen, steps_per_epoch=trGen.numSteps, 
        validation_data=cvGen, validation_steps=cvGen.numSteps,
        epochs=learning['minEpoch']-1, verbose=2)]
try:
    m.save (exp + '/cnn.h5', overwrite=True)
except TypeError:
    m.save_weights (exp + '/dnn.weights.h5', overwrite=True)
sys.stdout.flush()
sys.stderr.flush()

valErrorDiff = 1 + learning['minValError'] ## Initialise

## Continue training till validation loss stagnates
while learning['lrScaleCount']:
    print ('Learning rate: %f' % learning['rate'])
    h.append (m.fit_generator (trGen, steps_per_epoch=trGen.numSteps,
            validation_data=cvGen, validation_steps=cvGen.numSteps,
            epochs=1, verbose=2))
    try:
        m.save (exp + '/cnn.h5', overwrite=True)
    except TypeError:
        m.save_weights (exp + '/dnn.weights.h5', overwrite=True)
    sys.stdout.flush()
    sys.stderr.flush()

    ## Check validation error and reduce learning rate if required
    valErrorDiff = h[-2].history['val_loss'][-1] - h[-1].history['val_loss'][-1]
    if valErrorDiff < learning['minValError']:
        learning['rate'] *= learning['lrScale']
        learning['lrScaleCount'] -= 1
        K.set_value(m.optimizer.lr, learning['rate'])

