#!/bin/bash

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

def model_architecture (arch, inputFeatDim=None, outputFeatDim=None):
    ## Set dimensions
    if inputFeatDim == None:
        inputFeatDim = 4000
    if outputFeatDim == None:
        outputFeatDim = 144

    ## Else-if on architectures
    if arch == 'sep1D':
        ## Construct input and output
        x = keras.layers.Input (shape=(inputFeatDim,))
        y = keras.layers.Reshape((inputFeatDim, 1)) (x)

        ## First layer
        y = keras.layers.Conv1D(filters=80, kernel_size=30, strides=10) (y)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## Second layer
        y = keras.layers.Conv1D(filters=60, kernel_size=1, strides=1) (y)
        a,b,c = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,i])(y) for i in range(c)]
        a,b = [i.value for i in yl[0].shape]
        yl = [keras.layers.Reshape((b,1))(l) for l in yl]
        y = [keras.layers.Conv1D(filters=1, kernel_size=7, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (y, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)
        
        ## Third layer
        y = keras.layers.Conv1D(filters=60, kernel_size=1, strides=1) (y)
        a,b,c = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,i])(y) for i in range(c)]
        a,b = [i.value for i in yl[0].shape]
        yl = [keras.layers.Reshape((b,1))(l) for l in yl]
        y = [keras.layers.Conv1D(filters=1, kernel_size=7, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (y, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## MLP
        y = keras.layers.Flatten() (y)
        y = keras.layers.Dense(1024) (y)
        y = keras.layers.Activation('relu') (y)
        y = keras.layers.Dense(outputFeatDim) (y)
        y = keras.layers.Activation('softmax') (y)
        ## Define the model
        m = keras.models.Model (inputs=x, outputs=y)

    elif arch == 'sep1DRev':
        ## Construct input and output
        x = keras.layers.Input (shape=(inputFeatDim,))
        y = keras.layers.Reshape((inputFeatDim, 1)) (x)

        ## First layer
        y = keras.layers.Conv1D(filters=80, kernel_size=30, strides=10) (y)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## Second layer
        a,b,c = [i.value for i in y.shape]
        y = keras.layers.Reshape((b,c,1)) (y)
        y = keras.layers.Conv2D(filters=30, kernel_size=(7,1), strides=(1,1)) (y)
        a,b,c,d = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,:,i])(y) for i in range(d)]
        yl = [keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (yl, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)
        
        ## Third layer 
        a,b,c = [i.value for i in y.shape] 
        y = keras.layers.Reshape((b,c,1)) (y)
        y = keras.layers.Conv2D(filters=30, kernel_size=(7,1), strides=(1,1)) (y)
        a,b,c,d = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,:,i])(y) for i in range(d)]
        yl = [keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (yl, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## MLP
        y = keras.layers.Flatten() (y)
        y = keras.layers.Dense(1024) (y)
        y = keras.layers.Activation('relu') (y)
        y = keras.layers.Dense(outputFeatDim) (y)
        y = keras.layers.Activation('softmax') (y)
        ## Define the model
        m = keras.models.Model (inputs=x, outputs=y)

    elif arch == 'sep2D':
        rank = 2
        ## Construct input and output
        x = keras.layers.Input (shape=(inputFeatDim,))
        y = keras.layers.Reshape((inputFeatDim, 1)) (x)

        ## First layer
        y = keras.layers.Conv1D(filters=80, kernel_size=30, strides=10) (y)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## Second layer
        y = keras.layers.Conv1D(filters=rank*60, kernel_size=1, strides=1) (y)
        a,b,c = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,i:i+rank])(y) for i in range(0,c-rank+1,rank)]
        y = [keras.layers.Conv1D(filters=1, kernel_size=7, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (y, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)
        
        ## Third layer
        y = keras.layers.Conv1D(filters=rank*60, kernel_size=1, strides=1) (y)
        a,b,c = [i.value for i in y.shape]
        yl = [keras.layers.Lambda(lambda x: x[:,:,i:i+rank])(y) for i in range(0,c-rank+1,rank)]
        y = [keras.layers.Conv1D(filters=1, kernel_size=7, strides=1)(l) for l in yl]
        y = keras.layers.concatenate (y, axis=-1)
        y = keras.layers.pooling.MaxPooling1D(3) (y)
        y = keras.layers.Activation('relu') (y)

        ## MLP
        y = keras.layers.Flatten() (y)
        y = keras.layers.Dense(1024) (y)
        y = keras.layers.Activation('relu') (y)
        y = keras.layers.Dense(outputFeatDim) (y)
        y = keras.layers.Activation('softmax') (y)
        ## Define the model
        m = keras.models.Model (inputs=x, outputs=y)

    elif arch in ["3conv_1h", "cnn3"]:
        m = keras.models.Sequential([
                keras.layers.Reshape ((inputFeatDim, 1), input_shape=(inputFeatDim,)),
                keras.layers.Conv1D(80, 30, strides=10),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.Conv1D(60, 7),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.Conv1D(60, 7),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(1024),
                keras.layers.Activation('relu'),
                keras.layers.Dense(outputFeatDim),
                keras.layers.Activation('softmax')])

    elif arch == "separableConvolution":
        m = keras.models.Sequential([
                keras.layers.Reshape ((inputFeatDim, 1), input_shape=(inputFeatDim,)),
                keras.layers.Conv1D(80, 30, strides=10), 
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.SeparableConv1D(60, 7),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.SeparableConv1D(60, 7),
                keras.layers.pooling.MaxPooling1D(3),
                keras.layers.Activation('relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(1024),
                keras.layers.Activation('relu'),
                keras.layers.Dense(outputFeatDim),
                keras.layers.Activation('softmax')])

    else:
        TypeError ('Unknown architecture: ' + arch)
    
    ## Return the model
    return m 
