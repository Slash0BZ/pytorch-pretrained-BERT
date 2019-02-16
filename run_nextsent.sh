#!/usr/bin/env bash

pip install --editable .
python examples/run_classifier_nextsent.py \
  --task_name TEMPORAL \
  --do_eval \
  --do_lower_case \
  --data_dir samples/next_sent_sample.txt \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --output_dir tmp