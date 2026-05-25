CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" \
python ../src/finetune_bert.py -c configs_aiml/bert_aiml_epoch1.json
