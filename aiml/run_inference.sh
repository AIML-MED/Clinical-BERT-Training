#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
python ../src/run_bert_inference.py -c configs_aiml/bert_aiml.json
