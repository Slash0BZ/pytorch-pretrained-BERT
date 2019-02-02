#!/usr/bin/env bash

pip install --editable .
python examples/run_lm_finetuning.py \
  --bert_model bert-base-uncased \
  --do_train \
  --train_file samples/sample_text.txt \
  --output_dir models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128