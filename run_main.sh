#!/usr/bin/env bash

pip install --editable .
python examples/run_lm_finetuning.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file samples/dummy_data_small.txt \
  --output_dir models_lock \
  --num_train_epochs 100.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128