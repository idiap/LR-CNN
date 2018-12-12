#!/usr/bin/python3

## Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
## Written by S. Pavankumar Dubagunta <pavankumar [dot] dubagunta [at] idiap [dot] ch>,
## Vinayak Abrol and Mathew Magimai Doss <mathew [at] idiap [dot] ch>.
## 
## This file is part of LR-CNN.
## 
## LR-CNN is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License version 3 as
## published by the Free Software Foundation.
## 
## LR-CNN is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with LR-CNN. If not, see <http://www.gnu.org/licenses/>.


import sys
import numpy
import keras
import kaldiIO
from model_architecture import model_architecture
from signal import signal, SIGPIPE, SIG_DFL

if __name__ == '__main__':
    model = sys.argv[1]
    priors = sys.argv[2]
    arch = sys.argv[3]

    if not model.endswith('.h5'):
        raise TypeError ('Unsupported model type. Please use h5 format. Update Keras if needed')

    ## Load priors
    p = numpy.genfromtxt (priors, delimiter=',')
    p[p==0] = 1e-5 ## Deal with zero priors

    ## Load model
    m = model_architecture (arch, 4000, len(p))
    m.load_weights(model)

    arkIn = sys.stdin.buffer
    arkOut = sys.stdout.buffer
    encoding = sys.stdout.encoding
    signal (SIGPIPE, SIG_DFL)

    uttId, featMat = kaldiIO.readUtterance(arkIn)
    while uttId:
        ## Normalise features
        featMat = ((featMat.T - featMat.mean(axis=-1))/featMat.std(axis=-1)).T

        logProbMat = numpy.log (m.predict (featMat) / p)
        logProbMat [logProbMat == -numpy.inf] = -100
        
        kaldiIO.writeUtterance(uttId, logProbMat, arkOut, encoding)
        uttId, featMat = kaldiIO.readUtterance(arkIn)
    
