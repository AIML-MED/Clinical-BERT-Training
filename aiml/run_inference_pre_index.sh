#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
python ../src/run_bert_inference.py -c configs_aiml/bert_aiml_epoch1.json
