#!/bin/bash
sup_size=$1
vocab="vocab.freq5.bin"
if [ $sup_size != 'all' ]; then
train_file="train.$sup_size.bin"
else
train_file="train.bin"
fi
dropout=0.3
hidden_size=256
embed_size=128
ptrnet_hidden_dim=32
lr_decay=0.5
lstm='lstm'
model_name=model.sup.decoder.${lstm}.dropout${dropout}.freq5.${train_file}

python exp.py \
    --cuda \
    --mode train_decoder \
    --batch_size 10 \
    --train_file ./data/django/${train_file} \
    --dev_file ./data/django/dev.bin \
    --vocab ./data/django/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --ptrnet_hidden_dim ${ptrnet_hidden_dim} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>&1 | tee -a logs/${model_name}.log
