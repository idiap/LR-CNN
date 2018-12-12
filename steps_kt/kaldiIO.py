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


import numpy
import struct

## Read utterance
def readUtterance (ark):
    ## Read utterance ID
    uttId = b''
    c = ark.read(1)
    if not c:
        return None, None
    while c != b' ':
        uttId += c
        c = ark.read(1)
    ## Read feature matrix
    header = struct.unpack('<xcccc', ark.read(5))
    m, rows = struct.unpack('<bi', ark.read(5))
    n, cols = struct.unpack('<bi', ark.read(5))
    featMat = numpy.frombuffer(ark.read(rows * cols * 4), dtype=numpy.float32)
    return uttId.decode(), featMat.reshape((rows,cols))

def writeUtterance (uttId, featMat, ark, encoding):
    featMat = numpy.asarray (featMat, dtype=numpy.float32)
    m,n = featMat.shape
    ## Write header
    ark.write (struct.pack('<%ds'%(len(uttId)),uttId.encode(encoding)))
    ark.write (struct.pack('<cxcccc',' '.encode(encoding),'B'.encode(encoding),
                'F'.encode(encoding),'M'.encode(encoding),' '.encode(encoding)))
    ark.write (struct.pack('<bi', 4, m))
    ark.write (struct.pack('<bi', 4, n))
    ## Write feature matrix
    ark.write (featMat)

