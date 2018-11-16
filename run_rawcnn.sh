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


set -e

nj=10
. cmd.sh
. path.sh

## Add CUDA libraries to path if needed
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

## Configurable directories
train=data/train_raw    ## Data directories containing raw speech as features
dev=data/dev_raw
test=data/test_raw
trainmfcc=data/train    ## Data directories containing MFCC features
devmfcc=data/dev
lang=data/lang
fmllr=exp/tri3          ## FMLLR directory
gmm=exp/sgmm2_4         ## SGMM directory
arch="$1"   ## Architecture: see steps_kt/model_architecture.py for valid options.
            ## E.g. sep1D, sep2D.
exp=exp/rawcnn_$arch   ## Output directory.
init="$2"   ## Optional CNN directory for initialisation.

[[ -z $arch ]] && echo "Provide model architecture as argument" && exit 1
[[ ! -z $init ]] && initCNN=$init/cnn.h5 && initBaseName=$(basename $init)

## We already have dev set. So use it to cross-validate, and the entire training set to train.
[ -d ${train}_tr95 ] || ln -s train_raw ${train}_tr95
[ -d ${trainmfcc}_tr95 ] || ln -s train ${trainmfcc}_tr95
[ -d ${train}_cv05 ] || ln -s dev_raw ${train}_cv05
[ -d ${trainmfcc}_cv05 ] || ln -s dev ${trainmfcc}_cv05

## Align data using GMM
for dset in cv05 tr95; do
    [ -f ${fmllr}_ali_$dset/ali.1.gz ] || steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
        ${trainmfcc}_$dset $lang $fmllr ${fmllr}_ali_$dset
    [ -f ${gmm}_ali_$dset/ali.1.gz ] || steps/align_sgmm2.sh --nj $nj --cmd "$train_cmd" \
    --transform-dir ${fmllr}_ali_$dset ${trainmfcc}_$dset $lang $gmm ${gmm}_ali_$dset
done

mkdir -p $exp

## Copy transition model and tree
copy-transition-model $gmm/final.mdl $exp/final.mdl
copy-tree $gmm/tree $exp/tree

## Compute priors
python3 steps_kt/compute_priors.py $exp ${gmm}_ali_tr95 ${gmm}_ali_cv05

## Train
[ -f $exp/cnn.h5 ] || $python_cmd logs/rawcnn_${arch}_${initBaseName}_1.log \
    python3 steps_kt/train_rawcnn.py \
    ${train}_cv05 ${gmm}_ali_cv05 ${train}_tr95 ${gmm}_ali_tr95 $gmm $arch $exp $initCNN

## Make graph
[ -f $gmm/graph/HCLG.fst ] || \
    utils/mkgraph.sh ${lang}_test_bg $gmm $gmm/graph

## Decode
[ -f $exp/decode_bg_dev/lat.1.gz ] || bash steps_kt/decode_norm_arch.sh --nj $nj --arch "$arch" \
    --add-deltas "false" --norm-vars "false" --splice-opts "--left-context=12 --right-context=12" \
    $dev $gmm/graph $exp $exp/decode_bg_dev &
[ -f $exp/decode_bg_test/lat.1.gz ] || bash steps_kt/decode_norm_arch.sh --nj $nj --arch "$arch" \
    --add-deltas "false" --norm-vars "false" --splice-opts "--left-context=12 --right-context=12" \
    $test $gmm/graph $exp $exp/decode_bg_test &

wait

#### Align
##    [ -f ${exp}_ali ] || steps_kt/align.sh --nj $nj --cmd "$train_cmd" \
##        --add-deltas "true" --norm-vars "true" --splice-opts "--left-context=5 --right-context=5" \
##        $train $lang $exp ${exp}_ali
