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


## Begin configuration section
stage=0
nj=10
cmd=queue.pl

max_active=7000 # max-active
beam=15.0 # beam used
latbeam=8.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices
arch=""

skip_scoring=false # whether to skip WER scoring
scoring_opts=

splice_opts="--left-context=12 --right-context=12"
norm_vars="false"
add_deltas="false"

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: decode.sh [options] <data-dir> <graph-dir> <dnn-dir> <decode-dir>"
   echo " e.g.: decode.sh data/test exp/tri2b/graph exp/dnn_5a exp/dnn_5a/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --arch <architecture>                    # model architecture"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   exit 1;
fi

data=$1
graphdir=$2
dnndir=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.

dnn=$dnndir/cnn.h5
priors=$dnndir/priors.csv
[[ -f $dnndir/cnn.weights.h5 ]] && dnn=$dnndir/cnn.weights.h5

sdata=$data/split$nj;

[ -z "$arch" ] && echo "Model architecture not specified" && exit 1
mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $dnndir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp $dnndir/tree $dnn; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk --norm-vars=$norm_vars scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$splice_feats && feats="$feats splice-feats $splice_opts ark:- ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
feats="$feats steps_kt/nnet-forward-norm-arch.py $dnn $priors $arch |"

$decode_cmd JOB=1:$nj $dir/log/decode.JOB.log \
  latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $dnndir/final.mdl $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz"

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$decode_cmd" $data $graphdir $dir
fi

exit 0;
