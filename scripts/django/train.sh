#!/bin/bash
sup_size=$1
seed=$2
vocab="vocab.freq5.bin"
if [ $sup_size != 'all' ]; then
train_file="train.$sup_size.bin"
else
train_file="train.bin"
fi
dropout=0.2
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr_decay=0.5
beam_size=15
lstm='lstm_with_dropout'
model_name=model.sup.${lstm}.dropout${dropout}.freq5.${train_file}.seed$seed

python exp.py \
    --seed $seed \
    --cuda \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --train_file ./data/django/${train_file} \
    --dev_file ./data/django/dev.bin \
    --vocab ./data/django/${vocab} \
    --lstm ${lstm} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/${model_name} | tee -a logs/${model_name}.log

. scripts/django/test.sh saved_models/${model_name}.bin | tee -a logs/${model_name}.log
