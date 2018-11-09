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


from subprocess import Popen, PIPE, DEVNULL
import tempfile
import kaldiIO
import pickle
import shutil
import numpy
import os
##import ipdb

## Data generator class for Kaldi
class dataGenerator:
    def __init__ (self, data, ali, exp, batchSize=256, context=12):
        self.data = data
        self.ali = ali
        self.exp = exp
        self.batchSize = batchSize
        self.context = str(context)
        
        self.maxSplitDataSize = 100 ## These many utterances are loaded into memory at once.
        self.labelDir = tempfile.TemporaryDirectory()
        aliPdf = self.labelDir.name + '/alipdf.txt'
 
        ## Generate pdf indices
        Popen (['ali-to-pdf', exp + '/final.mdl',
                    'ark:gunzip -c %s/ali.*.gz |' % ali,
                    'ark,t:' + aliPdf]).communicate()

        ## Read labels
        with open (aliPdf) as f:
            labels, self.numFeats = self.readLabels (f)

        ## Determine the number of steps
        self.numSteps = -(-self.numFeats//self.batchSize)
      
        self.inputFeatDim = (2*context+1)*160 ## IMPORTANT: HARDCODED. Change if necessary.
        self.outputFeatDim = self.readOutputFeatDim()
        self.splitDataCounter = 0
        numpy.random.seed(512)
        
        self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        self.y = numpy.empty (0, dtype=numpy.int32)
        self.batchPointer = 0
        self.doUpdateSplit = True

        ## Read number of utterances
        with open (data + '/utt2spk') as f:
            self.numUtterances = sum(1 for line in f)
        self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)

        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        if os.path.isdir (data + 'split' + str(self.numSplit)):
            shutil.rmtree (data + 'split' + str(self.numSplit))
        Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()

        ## Save split labels and delete labels
        self.splitSaveLabels (labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()
        
    ## Determine the number of output labels
    def readOutputFeatDim (self):
        p1 = Popen (['am-info', '%s/final.mdl' % self.exp], stdout=PIPE)
        modelInfo = p1.stdout.read().splitlines()
        for line in modelInfo:
            if b'number of pdfs' in line:
                return int(line.split()[-1])

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1
            labels[line[0]] = numpy.array([int(i) for i in line[1:]], dtype=numpy.int32)
        return labels, numFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + 'utt/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)

    ## Return a batch to work on
    def getNextSplitData (self):
        p = Popen (['splice-feats','--print-args=false', '--left-context='+self.context, '--right-context='+self.context,
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/feats.scp',
                'ark:-'], stdout=PIPE)

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
    
        ##ipdb.set_trace()
        featList = []
        labelList = []
        while True:
            uid, featMat = kaldiIO.readUtterance (p.stdout)
            if uid == None:
                return (numpy.vstack(featList), numpy.hstack(labelList))
            if uid in labels:
                ## NOTE: The featMat may contain more frames if the labels were generated from
                ## MFCCs extracted on 25ms frames (and here we are using 10ms frames). Using
                ## these labels is fairly OK because the frame rate (10ms) is unchanged. We
                ## discard the last few frames that have no corresponding labels.
                featList.append (featMat[0:len(labels[uid])])
                labelList.append (labels[uid])

    ## Make the object iterable
    def __iter__ (self):
        return self

    ## Retrive a mini batch
    def __next__ (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break

            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
             
            ## Normalise means and variances of data
            x = ((x.T - x.mean(axis=-1))/x.std(axis=-1)).T
            
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            ## Shuffle data
            randomInd = numpy.array(range(len(self.x)))
            numpy.random.shuffle(randomInd)
            self.x = self.x[randomInd]
            self.y = self.y[randomInd]

            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False
        
        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize

        return (xMini, yMini)
