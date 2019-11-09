#!/bin/bash
sup_size=$1
seed=$2
vocab="vocab.bin"
if [ $sup_size != 'all' ]; then
train_file="train.$sup_size.bin"
else
train_file="train.bin"
fi
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
lstm='lstm_with_dropout'
model_name=model.atis.sup.${lstm}.dropout${dropout}.${train_file}.seed$seed
mkdir -p saved_models
mkdir -p logs
python exp.py \
    --seed $seed \
    --cuda \
    --mode train \
    --lang lambda_dcs \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --train_file data/atis/${train_file} \
    --dev_file data/atis/dev.bin \
    --vocab data/atis/${vocab} \
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
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>&1 | tee -a logs/${model_name}.log

. scripts/atis/test.sh saved_models/${model_name}.bin 2>&1 | tee -a logs/${model_name}.log
