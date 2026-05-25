#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
python ../src/finetune_bert.py -c configs_aiml/bert_aiml_pre_index.json
