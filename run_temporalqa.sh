#!/usr/bin/env bash

pip install --editable .
python examples/run_classifier.py \
  --task_name TEMPORAL \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir samples/temporal_data_split \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./tmp