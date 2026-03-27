#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" \
python ../src/pretrain_bert.py -c configs_aiml/pretrain_aiml.yaml
