#!/bin/bash
sup_size=$1
seed=$2
if [ $sup_size != 'all' ]; then
train_data="train.$sup_size.bin"
unlabeled_data="train.$sup_size.remaining.bin"
encoder=model.sup.lstm_with_dropout.dropout0.2.freq5.train.$sup_size.bin.seed$seed.bin
decoder=model.sup.decoder.lstm.dropout0.3.freq5.train.$sup_size.bin.bin
else
train_data="train.bin"
unlabeled_data="train.bin"
encoder=model.sup.lstm_with_dropout.dropout0.2.freq5.train.bin.seed$seed.bin
decoder=model.sup.decoder.lstm.dropout0.3.freq5.train.bin.bin
fi
lr_decay=0.5
unsup_loss_weight=0.1
model_name=model.python.semisup.${encoder}.seed$seed

python exp.py \
  --seed $seed \
    --cuda \
    --mode train_semi_vae \
    --batch_size 10 \
    --train_file ./data/django/${train_data} \
    --unlabeled_file ./data/django/${unlabeled_data} \
    --dev_file ./data/django/dev.bin \
    --load_model saved_models/${encoder} \
    --load_decoder saved_models/${decoder} \
    --unsup_loss_weight 0.1 \
    --load_prior saved_models/prior/prior.dropout0.2.freq_cutoff5.src.bin \
    --load_src_lm saved_models/src_lm.dropout0.4.freq5.train.bin.bin \
    --patience 5 \
    --max_num_trial 5 \
    --lr_decay ${lr_decay} \
    --save_to saved_models/${model_name} 2>&1 | tee -a logs/${model_name}.log

. scripts/django/test.sh saved_models/${model_name}.encoder.bin 2>&1 | tee -a logs/${model_name}.log
    