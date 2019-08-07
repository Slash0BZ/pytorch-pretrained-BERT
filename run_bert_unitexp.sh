#!/usr/bin/env bash

pip install --editable .
python examples/run_bert_custom.py \
  --task_name TEMPORALVERB \
  --do_train \
  --do_lower_case \
  --data_dir samples/UD_English \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 1.0 \
  --output_dir ./bert_unitexp_0
