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


# Begin configuration section.  
nj=50
stage=0
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
cmd="run.pl"
arch=""

splice_opts="--left-context=12 --right-context=12"
norm_vars="false"
add_deltas="false"

align_to_lats=false # optionally produce alignment in lattice format
lats_decode_opts="--acoustic-scale=0.1 --beam=20 --lattice_beam=10"
lats_graph_scales="--transition-scale=1.0 --self-loop-scale=0.1"

use_gpu="no" # yes|no|optionaly
# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. cmd.sh
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --arch <architecture>                            # model architecture"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

[ -z "$splice_opts" ] && splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`

[ -z "$arch" ] && echo "Model architecture not specified" && exit 1

## Change nj to number of speakers if <50.
nspk=$(wc -l <$data/spk2utt)
[ $nj -le $nspk ] || nj=$nspk

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

cp $srcdir/{tree,final.mdl} $dir || exit 1;

# Select default locations to model files
nnet=$srcdir/cnn.h5
priors=$srcdir/priors.csv
model=$dir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/L.fst $nnet $model; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# PREPARE FEATURE EXTRACTION PIPELINE
# Create the feature stream:
## Set up the features
echo "$0: feature: splice(${splice_opts}) norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk --norm-vars=$norm_vars scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
feats="$feats splice-feats $splice_opts ark:- ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"

# Finally add feature_transform and the MLP
feats="$feats steps_kt/nnet-forward-norm-arch.py $nnet $priors $arch |"

echo "$0: aligning data '$data' using nnet/model '$srcdir', putting alignments in '$dir'"
# Map oovs in reference transcription 
tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
# We could just use align-mapped in the next line, but it's less efficient as it compiles the
# training graphs one by one.
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    align-compiled-mapped $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/final.mdl ark:- \
      "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

# Optionally align to lattice format (handy to get word alignment)
if [ "$align_to_lats" == "true" ]; then
  echo "$0: aligning also to lattices '$dir/lat.*.gz'"
  $cmd JOB=1:$nj $dir/log/align_lat.JOB.log \
    compile-train-graphs $lat_graph_scale $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    latgen-faster-mapped $lat_decode_opts --word-symbol-table=$lang/words.txt $dir/final.mdl ark:- \
      "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

echo "$0: done aligning data."
