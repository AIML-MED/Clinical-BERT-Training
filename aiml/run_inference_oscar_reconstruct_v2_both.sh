#!/usr/bin/env bash

set -e

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python ../src/run_bert_inference.py -c configs_aiml/bert_oscar_reconstruct_v2_balanced_test.json

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python ../src/run_bert_inference.py -c configs_aiml/bert_oscar_reconstruct_v2_naturalprevalence_test.json
