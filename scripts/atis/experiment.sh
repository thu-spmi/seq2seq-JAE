#!/bin/bash
sup_size=$1
seed=$2
PYTHONPATH=. python scripts/atis/train_src_lm.sh #train p(x) model, could be shared for different parallel data size
PYTHONPATH=. python scripts/atis/train_prior.sh $sup_size #train p(z) model
PYTHONPATH=. python scripts/atis/train_decoder.sh $sup_size # supervisedly train decoder
PYTHONPATH=. python scripts/atis/train.sh $sup_size $seed # supervisedly train encoder
PYTHONPATH=. python scripts/atis/train_semi_vae.sh $sup_size $seed # semi-supervised VAE
PYTHONPATH=. python scripts/atis/train_semi_jae.sh $sup_size $seed 0 # semi-supervised JAE
PYTHONPATH=. python scripts/atis/train_semi_jae.sh $sup_size $seed 1 # semi-supervised bi-JAE


    