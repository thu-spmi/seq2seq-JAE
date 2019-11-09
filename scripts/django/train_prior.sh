#!/bin/bash

code_dir="data/django_code"
dropout=0.2
freq_cutoff=5
model_name=prior.dropout${dropout}.freq_cutoff${freq_cutoff}

PYTHONPATH=. python scripts/django/train_lstm_prior.py \
    --cuda \
    --batch_size 32 \
    --code_dir ${code_dir} \
    --dropout ${dropout} \
    --freq_cutoff ${freq_cutoff} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay 0.5 \
    --log_every 10 \
    --save_to saved_models/prior/${model_name}