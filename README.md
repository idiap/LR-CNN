# Low-Rank CNN

Trains low-rank CNNs from raw speech using Keras/Tensorflow,
with inputs from Kaldi directories.

## Features

1. Trains CNNs from Kaldi GMM system

2. Works with standard Kaldi data and alignment directories

3. Decodes test utterances in Kaldi style

## Dependencies

1. Python 3.4+

2. Keras with Tensorflow/Theano backend

3. Kaldi (obtained using git)

## Using the Code

1. Train a GMM system in Kaldi.

2. Place steps\_kt and run\_\*.sh in the working directory.

3. Apply the patch compute-raw-feats.patch to Kaldi. To do this:
    ```
    . ./path.sh ## To get $KALDI_ROOT environment variable.
    mv compute-raw-feats.patch $KALDI_ROOT/
    cd $KALDI_ROOT/
    git apply compute-raw-feats.patch
    cd src/
    make depend ## [-j 4]
    make ## [-j 4]
    ```
    Note: This creates a new executable compute-raw-feats in
    src/featbin/ directory of Kaldi. It does not alter any of
    the existing Kaldi tools.

4. Extract raw features using extract.sh.

5. Configure and run run\_\*.sh.

## Code Components

1. train\*\_rawcnn.py is the Keras training script.

2. Model architecture can be configured in model\_architecture.py.

2. dataGeneratorSRaw.py provides an object that reads Kaldi data and 
  alignment directories in batches and retrieves mini-batches for 
  training.

3. nnet-forward-norm-arch.py passes test features through the trained
  CNNs and outputs log posterior probabilities in Kaldi format.

4. kaldiIO.py reads and writes Kaldi-type binary features.

5. decode\_norm\_arch.py is the decoding script.

6. align\_norm\_arch.sh is the alignment script.

7. compute\_priors.py computes priors.

## Training Schedule

The script uses stochastic gradient descent with 0.5 momentum. It 
starts with a learning rate of 0.1 for a minimum of 5 
epochs. Whenever the validation loss reduces by less than 0.002 
between successive epochs, the learning rate is halved. Halving
is performed for a total of 18 times.

## Contributors

Idiap Research Institute

Authors: S. Pavankumar Dubagunta and Vinayak Abrol

## License

GNU GPL v3
