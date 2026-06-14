#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python ../src/finetune_bert.py -c configs_aiml/bert_oscar_reconstruct_v2.json
