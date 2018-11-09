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

