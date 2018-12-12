#!/bin/bash

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


nj=10
. ./cmd.sh
. ./path.sh

rawdir=raw      ## Directories to store raw features in ark format

for x in train dev test; do
  [[ ! -d data/$x ]] && echo "Expected the directory data/$x to exist" && exit 1
  mkdir data/${x}_raw
  cp data/$x/* data/${x}_raw/
  rm -f data/${x}_raw/feats.scp data/${x}_raw/cmvn.scp
  
  ## Extract raw features
  steps_kt/make_raw.sh --cmd "$train_cmd" --nj $nj data/${x}_raw exp/make_raw/${x}_raw $rawdir

  ## Compute fake CMVN stats
  steps/compute_cmvn_stats.sh --fake data/${x}_raw exp/make_raw/${x}_raw $rawdir
done

