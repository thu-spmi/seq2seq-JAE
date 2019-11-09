#!/bin/bash

vocab="vocab.freq5.bin"
train_file="train.bin"
dropout=0.4
hidden_size=256
embed_size=128
lr_decay=0.5
model_name=src_lm.dropout${dropout}.freq5.${train_file}

PYTHONPATH=. python scripts/django/train_source_lm.py \
    --cuda \
    --batch_size 10 \
    --train_file data/django/${train_file} \
    --dev_file data/django/dev.bin \
    --vocab data/django/${vocab} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --log_every 50 \
    --save_to saved_models/${model_name}
