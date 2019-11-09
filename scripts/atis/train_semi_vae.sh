#!/bin/bash
sup_size=$1
seed=$2

vocab="vocab.bin"
if [ $sup_size != 'all' ]; then
train_data="train.$sup_size.bin"
unlabeled_data="train.$sup_size.remaining.bin"
prior=saved_models/prior/prior.train.$sup_size.bin.dropout0.2.bin
encoder=model.atis.sup.lstm_with_dropout.dropout0.3.train.$sup_size.bin.seed$seed.bin
decoder=model.atis.sup.decoder.dropout0.3.train.$sup_size.bin.bin
else
train_data="train.bin"
unlabeled_data="train.bin"
prior=saved_models/prior/prior.train.bin.dropout0.2.bin
encoder=model.atis.sup.lstm_with_dropout.dropout0.3.train.bin.seed$seed.bin
decoder=model.atis.sup.decoder.dropout0.3.train.bin.bin
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
model_name=model.atis.semi.${lstm}.dropout${dropout}.${train_data}.seed$seed
mkdir -p saved_models
mkdir -p logs
python exp.py \
    --seed $seed \
    --cuda \
    --mode train_semi_vae \
    --lang lambda_dcs \
    --batch_size 10 \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --train_file data/atis/${train_data} \
    --unlabeled_file data/atis/${unlabeled_data} \
    --dev_file data/atis/dev.bin \
    --vocab data/atis/${vocab} \
    --load_model saved_models/${encoder} \
    --load_decoder saved_models/${decoder} \
    --lstm ${lstm} \
    --load_prior $prior \
    --load_src_lm saved_models/src_lm.atis.hidden256.embed128.dropout0.3.lr_decay0.5.vocab.bin.train.bin.bin \
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
        --unsup_loss_weight 0.1 \
    --log_every 50 \
    --save_to saved_models/${model_name} 2>&1 | tee -a logs/${model_name}.log

. scripts/atis/test.sh saved_models/${model_name}.encoder.bin 2>&1 | tee -a logs/${model_name}.log
