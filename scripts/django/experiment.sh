#!/bin/bash
sup_size=$1
seed=$2
PYTHONPATH=. python scripts/django/train_src_lm.sh #train p(x) model, could be shared for different parallel data size
PYTHONPATH=. python scripts/django/train_prior.sh #train p(z) model, could be shared for different parallel data size, the django source code should be placed at data/django_code
PYTHONPATH=. python scripts/django/train_decoder.sh $sup_size # supervisedly train decoder
PYTHONPATH=. python scripts/django/train.sh $sup_size $seed # supervisedly train encoder
PYTHONPATH=. python scripts/django/train_semi_vae.sh $sup_size $seed # semi-supervised VAE
PYTHONPATH=. python scripts/django/train_semi_jae.sh $sup_size $seed 0 # semi-supervised JAE
PYTHONPATH=. python scripts/django/train_semi_jae.sh $sup_size $seed 1 # semi-supervised bi-JAE


    